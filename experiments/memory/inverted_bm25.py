"""Conventional retained BM25 postings for the corpus-size ladder.

The formula/tokenizer match compressed_routing_map.BM25. This changes the
implementation, not retrieval semantics or evidence authority. No model,
training, source reads, network access or index construction occurs on import.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.memory.compressed_routing_map import lexical_tokens


class InvertedBM25:
    """Build once, visit only query-term postings, return original row scores.

    Returned score vectors still allocate O(documents) bytes. Vocabulary lookup
    uses a Python dictionary; nbytes reports numeric payload, NOT resident memory.
    Query terms are a set, as in the reference. Floating summation order across
    different Python hash seeds is not a bit-exact cross-process contract.
    """

    def __init__(self, texts):
        postings = defaultdict(list)
        lengths = []
        for index, text in enumerate(texts):
            counts = Counter(lexical_tokens(text))
            lengths.append(sum(counts.values()))
            for word, frequency in counts.items():
                postings[word].append((index, frequency))
        vocabulary = tuple(sorted(postings))
        offsets = [0]
        documents, frequencies = [], []
        for word in vocabulary:
            for index, frequency in postings[word]:
                documents.append(index)
                frequencies.append(frequency)
            offsets.append(len(documents))
        self._initialize(vocabulary, np.asarray(offsets, dtype=np.int64),
                         np.asarray(documents, dtype=np.int64),
                         np.asarray(frequencies, dtype=np.float32),
                         np.asarray(lengths, dtype=np.int64))

    def _initialize(self, vocabulary, offsets, documents, frequencies, lengths):
        if (not all(isinstance(w, str) and w for w in vocabulary)
                or tuple(sorted(set(vocabulary))) != tuple(vocabulary)
                or offsets.dtype != np.int64 or documents.dtype != np.int64
                or lengths.dtype != np.int64 or frequencies.dtype != np.float32
                or any(x.ndim != 1 for x in (offsets, documents, frequencies, lengths))
                or len(offsets) != len(vocabulary) + 1 or offsets[0] != 0
                or offsets[-1] != len(documents) or len(documents) != len(frequencies)
                or (np.diff(offsets) <= 0).any() or (lengths < 0).any()
                or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
                or (frequencies != np.floor(frequencies)).any()
                or (documents < 0).any() or (documents >= len(lengths)).any()):
            raise ValueError("invalid inverted BM25 state")
        totals = np.zeros(len(lengths), dtype=np.float64)
        for a, b in zip(offsets[:-1], offsets[1:]):
            ids = documents[a:b]
            if (np.diff(ids) <= 0).any():
                raise ValueError("posting document IDs must strictly increase")
            totals[ids] += frequencies[a:b]
        if not np.array_equal(totals, lengths):
            raise ValueError("posting frequencies and document lengths disagree")
        self.vocabulary = tuple(vocabulary)
        self.lookup = MappingProxyType({w: i for i, w in enumerate(vocabulary)})
        self.offsets, self.documents = offsets, documents
        self.frequencies, self.lengths = frequencies, lengths
        self.average = max(float(lengths.mean()), 1.) if len(lengths) else 1.
        self.norm = 1.5 * (.25 + .75 * lengths / self.average)
        for array in (offsets, documents, frequencies, lengths, self.norm):
            array.flags.writeable = False

    @property
    def nbytes(self):
        """Numeric serving payload only; excludes Python vocabulary and heap."""
        return sum(a.nbytes for a in (self.offsets, self.documents,
                                     self.frequencies, self.lengths, self.norm))

    def score(self, query):
        out = np.zeros(len(self.lengths), dtype=np.float32)
        for word in set(lexical_tokens(query)):
            position = self.lookup.get(word)
            if position is None:
                continue
            a, b = self.offsets[position:position + 2]
            ids, frequency = self.documents[a:b], self.frequencies[a:b]
            df = int(b - a)
            idf = math.log(1 + (len(self.lengths) - df + .5) / (df + .5))
            out[ids] += idf * frequency * 2.5 / (frequency + self.norm[ids])
        return out

    def query_work(self, query):
        """Logical postings visited, not bytes physically read or model work."""
        positions = [self.lookup[w] for w in set(lexical_tokens(query))
                     if w in self.lookup]
        return dict(
            matching_terms=len(positions),
            postings_visited=sum(int(self.offsets[i + 1] - self.offsets[i])
                                 for i in positions),
            score_vector_bytes=len(self.lengths) * np.dtype(np.float32).itemsize,
            reader_calls=0,
        )

    def save(self, path):
        """Create a new compact archive; never overwrite an existing index."""
        encoded = [word.encode("utf8") for word in self.vocabulary]
        word_offsets = np.cumsum([0, *map(len, encoded)], dtype=np.int64)
        with Path(path).open("xb") as output:
            np.savez(output, schema=np.asarray([1], dtype=np.int64),
                     word_bytes=np.frombuffer(b"".join(encoded), dtype=np.uint8),
                     word_offsets=word_offsets, offsets=self.offsets,
                     documents=self.documents, frequencies=self.frequencies,
                     lengths=self.lengths)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            expected = {"schema", "word_bytes", "word_offsets", "offsets",
                        "documents", "frequencies", "lengths"}
            if set(state.files) != expected or not np.array_equal(state["schema"], [1]):
                raise ValueError("unknown inverted BM25 archive")
            raw, starts = state["word_bytes"], state["word_offsets"]
            if (raw.dtype != np.uint8 or starts.dtype != np.int64
                    or raw.ndim != 1 or starts.ndim != 1 or len(starts) == 0
                    or starts[0] != 0 or starts[-1] != len(raw)
                    or (np.diff(starts) <= 0).any()):
                raise ValueError("invalid vocabulary encoding")
            vocabulary = tuple(raw[a:b].tobytes().decode("utf8")
                               for a, b in zip(starts[:-1], starts[1:]))
            result = cls.__new__(cls)
            result._initialize(vocabulary, *(state[name].copy() for name in
                ("offsets", "documents", "frequencies", "lengths")))
        return result
