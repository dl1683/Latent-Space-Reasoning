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


MAX_TOKENS = 120_000_000


def get_texts():
    from datasets import load_dataset
    ds = load_dataset(DS, CFG, split="train")
    return ds["text"]


def prepare(tag, max_tokens=120_000_000):
    """Vectorized: tokenize into one flat int32 array (doc breaks kept),
    bincount for freq, flattened pair-window bincount for co-occurrence."""
    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[tag])
    vocab_size = len(tok)
    print(tag, "tokenizing...", flush=True)
    docs = []
    ntok = 0
    for t in get_texts():
        if not t.strip():
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"]
        docs.append(np.asarray(ids, dtype=np.int32))
        ntok += len(ids)
        if ntok >= max_tokens:
            break
    # join with -1 separators (never a valid id)
    arrs, seps = [], 0
    for d_ in docs:
        arrs.append(d_); arrs.append(np.array([-1], dtype=np.int32)); seps += 1
    flat = np.concatenate(arrs)
    print(tag, "tokens", ntok, "docs", len(docs), flush=True)
    freq = np.bincount(flat[flat >= 0].astype(np.int64), minlength=vocab_size).astype(np.int64)
    top = np.argsort(freq)[::-1][:K]
    kept_mass = float(freq[top].sum() / max(freq.sum(), 1))
    remap = -np.ones(vocab_size, dtype=np.int32)
    remap[top] = np.arange(K, dtype=np.int32)
    print(tag, "kept mass", round(kept_mass, 4), flush=True)
    m = remap[flat]
    valid = m >= 0
    # pair offsets: within-doc only -> mask pairs crossing separators
    mi = m.copy()
    mi[~valid] = -1
    total_nnz = 0
    C = np.zeros((K, K), dtype=np.int64) if K <= 8192 else None
    # co-occurrence via bincount on flattened indices, per offset
    for off in range(1, WINDOW + 1):
        a = mi[:-off]; b = mi[off:]
        ok = (a >= 0) & (b >= 0)
        # doc-crossing guard: positions separated by a -1 are already excluded
        # because -1 maps to invalid; but tokens separated by ANOTHER token that
        # is invalid still belong to same doc pair window - acceptable (window
        # pairs span gaps; count them, matching the original doc-level intent
        # only when no separator lies between; we approximate by counting all
        # valid-valid pairs within off tokens INCLUDING across dropped rare
        # tokens - declared in prepare_{tag}.json)
        idx = a[ok].astype(np.int64) * K + b[ok].astype(np.int64)
        bc = np.bincount(idx, minlength=K * K)
        total_nnz += int((bc > 0).sum())
        C += bc.reshape(K, K)
        print(tag, f"offset {off} done", flush=True)
    import scipy.sparse as sp
    Cm = sp.csr_matrix(C)
    sp.save_npz(OUT / f"cooc_{tag}.npz", Cm)
    np.save(OUT / f"freq_{tag}.npy", freq[top])
    json.dump(dict(tag=tag, vocab=vocab_size, K=K, window=WINDOW, tokens=ntok,
                   docs=len(docs), kept_mass=kept_mass, nnz=int(Cm.nnz),
                   gap_note="pairs counted across dropped-rare-token gaps within window"),
              open(OUT / f"prepare_{tag}.json", "w"))
    print(tag, "DONE nnz", Cm.nnz, flush=True)


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
