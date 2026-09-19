"""HP-C1: spectral inheritance test (H2). theory/HD_POWER_LAW_ALLOCATION.md §8.

--prepare : download WikiText-103, tokenize per model, count +-2 window
            co-occurrences over top-K (8192) vocab, cache sparse counts.
--measure : NOT the frozen measurement until §8 is final-frozen; builds
            M-operator (sparse log part + two rank-1 terms), randomized SVD
            top-d, estimator battery, and compares to embedding spectra.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

OUT = Path(__file__).parent / "results" / "hpc1_inheritance"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"smollm2": "HuggingFaceTB/SmolLM2-135M",
          "qwen3": "Qwen/Qwen3-0.6B",
          "phi15": "microsoft/phi-1_5"}
K = 8192
WINDOW = 2
DS = "Salesforce/wikitext"
CFG = "wikitext-103-raw-v1"


def get_texts(max_tokens_hint=120_000_000):
    from datasets import load_dataset
    ds = load_dataset(DS, CFG, split="train")
    return ds["text"]


def prepare(tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[tag])
    texts = get_texts()
    vocab_size = len(tok)
    # frequency pass to select top-K ids
    print(tag, "vocab", vocab_size, "pass 1: frequencies")
    freq = np.zeros(vocab_size, dtype=np.int64)
    buf_ids = []
    ntok = 0
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"]
        buf_ids.append(np.asarray(ids, dtype=np.int32))
        np.add.at(freq, ids, 1)
        ntok += len(ids)
        if i % 20000 == 0:
            print(" doc", i, "tokens", ntok)
        if ntok >= max_tokens_hint:
            break
    top = np.argsort(freq)[::-1][:K]
    kept_mass = freq[top].sum() / max(freq.sum(), 1)
    remap = -np.ones(vocab_size, dtype=np.int32)
    remap[top] = np.arange(K, dtype=np.int32)
    print(tag, "kept mass", round(float(kept_mass), 4), "tokens", ntok)
    # co-occurrence pass over the SAME buffer (single corpus traversal)
    print(tag, "pass 2: co-occurrence")
    acc = None
    for ids in buf_ids:
        m = remap[ids]
        m = m[m >= 0]
        if len(m) < 2 * WINDOW + 1:
            continue
        rows, cols = [], []
        for off in range(1, WINDOW + 1):
            rows.append(m[:-off]); cols.append(m[off:])
            rows.append(m[off:]); cols.append(m[:-off])
        r = np.concatenate(rows); c = np.concatenate(cols)
        keep = r >= 0  # already filtered, belt and braces
        one = sp.coo_matrix((np.ones(r.shape[0], dtype=np.int32), (r, c)), shape=(K, K))
        acc = one if acc is None else acc + one
        if ntok and len(buf_ids) % 20000 == 0:
            print("  chunk", len(buf_ids))
    C = acc.tocsr()
    C.sum_duplicates()
    sp.save_npz(OUT / f"cooc_{tag}.npz", C)
    np.save(OUT / f"freq_{tag}.npy", freq[top])
    json.dump(dict(tag=tag, vocab=vocab_size, K=K, window=WINDOW, tokens=ntok,
                   kept_mass=float(kept_mass), nnz=int(C.nnz)),
              open(OUT / f"prepare_{tag}.json", "w"))
    print(tag, "done nnz", C.nnz)


def measure(tag, d, alphas=(0.1,)):
    """M = log(C + a) - log(c_i + a*K), restricted to top-K; operator form:
    sparse part on nonzeros + dense rank-2 corrections."""
    C = sp.load_npz(OUT / f"cooc_{tag}.npz").tocsr()
    meta = json.load(open(OUT / f"prepare_{tag}.json"))
    import scipy.sparse.linalg as spl
    for a in alphas:
        c = np.asarray(C.sum(axis=1)).ravel()
        coo = C.tocoo()
        Svals = np.log(coo.data + a)  # nonzero part of log(C+a)
        S = sp.coo_matrix((Svals, (coo.row, coo.col)), shape=C.shape).tocsr()
        # M_ij = S_ij (+ log a on zero entries) - log(c_i + aK) for ALL j
        # M = S + log(a)*ones@ones^T - k outer ones^T, k_i = log(c_i + aK)
        k = np.log(c + a * K)

        def mv(x):
            return S @ x + np.log(a) * x.sum() - k * x.sum()

        def rmv_exact(y):
            return S.T @ y + np.log(a) * y.sum() - k.dot(y) * np.ones(C.shape[1])

        op = spl.LinearOperator(C.shape, matvec=mv, rmatvec=rmv_exact)
        t0 = time.time()
        U, sv, Vt = spl.svds(op, k=min(d, K - 2), which="LM")
        sv = np.sort(sv)[::-1]
        lam = np.maximum(sv ** 2, 1e-300)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from hp_spectral_lib import battery
        bat = battery(lam)
        json.dump(dict(tag=tag, d=d, alpha_smooth=a, meta=meta,
                       singvals=sv.tolist(), battery=bat),
                  open(OUT / f"mspec_{tag}_d{d}_a{a}.json", "w"))
        print(tag, "d", d, "a", a, "alpha_top90", bat["top90"]["alpha"],
              "r2", bat["top90"]["r2_pl"], "wall", round(time.time() - t0, 1))


def embed_measure(tag, d):
    """Re-measure the model's input-embedding spectrum with the battery."""
    import torch, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hp_spectral_lib import battery
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODELS[tag], torch_dtype=torch.float32)
    E = m.get_input_embeddings().weight.detach().numpy()  # n x d
    lam = np.linalg.eigvalsh(E.T @ E)[::-1]
    lam = np.maximum(lam, 1e-300)
    bat = battery(lam)
    json.dump(dict(tag=tag, d=d, battery=bat, lam=lam.tolist()),
              open(OUT / f"embspec_{tag}_d{d}.json", "w"))
    print("emb", tag, bat["top90"]["alpha"], bat["top90"]["r2_pl"])
    del m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prepare", "measure", "embed"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--d", type=int, default=None)
    a = ap.parse_args()
    if a.phase == "prepare":
        prepare(a.tag)
    elif a.phase == "embed":
        dims = {"smollm2": 576, "qwen3": 1024, "phi15": 2048}
        embed_measure(a.tag, a.d or dims[a.tag])
    else:
        dims = {"smollm2": 576, "qwen3": 1024, "phi15": 2048}
        measure(a.tag, a.d or dims[a.tag])
