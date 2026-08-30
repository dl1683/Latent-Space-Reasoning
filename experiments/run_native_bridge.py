"""native_bridge_v1: first empirical test of the d_infinity bridge framework.

For each of 24 registered entities, does the correct-label residual centroid edit
bring the source prompt within the native target-fiber criterion and beat both no
edit and the wrong-label centroid? Surgeon-world artifact; no denizen bridge claimed.

    python experiments/run_native_bridge.py --config experiments/config/native_bridge_v1.json --stage smoke
    python experiments/run_native_bridge.py --config experiments/config/native_bridge_v1.json --stage science --lock-row experiments/results/native_bridge_v1/lock_row.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe

# --------------- sqrt-JS metric ---------------

def _normalize_logp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return x - (m + np.log(np.exp(x - m).sum(dtype=np.float64)))


def sqrt_js(log_p: np.ndarray, log_q: np.ndarray) -> float:
    if not (np.all(np.isfinite(log_p)) and np.all(np.isfinite(log_q))):
        raise ValueError("Non-finite input to sqrt_js")
    lp = _normalize_logp(log_p)
    lq = _normalize_logp(log_q)
    lm = np.logaddexp(lp, lq) - np.log(2.0)
    p = np.exp(lp)
    q = np.exp(lq)
    js = (np.sum(p * (lp - lm), dtype=np.float64)
          + np.sum(q * (lq - lm), dtype=np.float64)) / (2.0 * np.log(2.0))
    return float(np.sqrt(np.clip(js, 0.0, 1.0)))


def c9_from_full(log_full: np.ndarray, numeral_ids: list[int]) -> np.ndarray:
    probs = np.exp(_normalize_logp(log_full))
    num_probs = probs[numeral_ids]
    other = max(0.0, 1.0 - num_probs.sum())
    c9_probs = np.append(num_probs, other)
    c9_probs = np.maximum(c9_probs, 0.0)
    s = c9_probs.sum()
    if s > 0:
        c9_probs /= s
    return np.log(np.maximum(c9_probs, 1e-45))

# --------------- manifest and row loading ---------------

def verify_authorities(cfg: dict) -> dict[str, str]:
    shas = {}
    for name, info in cfg["authorities"].items():
        actual = hashlib.sha256(open(info["path"], "rb").read()).hexdigest()
        if actual != info["sha256"]:
            return None
        shas[name] = actual
    return shas


def build_row_manifest(cfg: dict, rows_data: list[dict]) -> dict | str:
    entities = cfg["entities"]
    rows = []
    for e in entities:
        i = e["id"]
        sr_idx = e["source_row"]
        sr = rows_data[sr_idx]
        if sr["e"] != i or sr["destroyed"]:
            return f"INVALID — ROW/TOKENIZATION MANIFEST: source row {sr_idx} entity {i}"
        if sr_idx != 128 * i + 16 * (i % 8):
            return f"INVALID — ROW/TOKENIZATION MANIFEST: source row formula mismatch entity {i}"
        rows.append({"entity_id": i, "role": "source", "row_idx": sr_idx,
                      "e": sr["e"], "s": sr.get("s"), "t": sr.get("t"),
                      "span": sr.get("span"), "len": sr.get("len")})
        for ti, tr in enumerate(e["target_rows"]):
            r = rows_data[tr]
            if r["e"] != i or r["destroyed"]:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: target row {tr} entity {i}"
            expected_tr = 128 * i + 16 * ((i + 1) % 8) + 4 * (ti + 1)
            if tr != expected_tr:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: target row formula mismatch entity {i} target {ti+1}"
            rows.append({"entity_id": i, "role": f"target_{ti+1}", "row_idx": tr,
                          "e": r["e"], "s": r.get("s"), "t": r.get("t"),
                          "span": r.get("span"), "len": r.get("len")})
        if e["target_label"] != (i + 1) % 8:
            return f"INVALID — ROW/TOKENIZATION MANIFEST: target_label mismatch entity {i}"
        if e["wrong_label"] != ((i + 1) % 8 + 1) % 8:
            return f"INVALID — ROW/TOKENIZATION MANIFEST: wrong_label mismatch entity {i}"
        for dr in e["correct_centroid_donor_rows"]:
            r = rows_data[dr]
            if r["destroyed"]:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: correct donor row {dr} destroyed"
            if r["e"] == i:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: correct donor row {dr} is own entity"
        for dr in e["wrong_centroid_donor_rows"]:
            r = rows_data[dr]
            if r["destroyed"]:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: wrong donor row {dr} destroyed"
            if r["e"] == i:
                return f"INVALID — ROW/TOKENIZATION MANIFEST: wrong donor row {dr} is own entity"
    assert len(rows) == len(entities) * 4, f"Manifest rows {len(rows)} != {len(entities) * 4}"
    manifest_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"rows": rows, "count": len(rows), "entities": len(entities), "hash": manifest_hash}

# --------------- call table enumeration ---------------

FAMILIES = ["target_1", "target_2", "target_3", "pasteback", "native", "source", "centroid", "wrong"]
WORDS = ["epsilon", "a_P", "a_Q", "a_Pa_P", "a_Pa_Q", "a_Qa_P", "a_Qa_Q"]


def _recipe_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


SMOKE_FAMILIES = ["source_plain", "source_hooked", "target_plain", "target_pasteback"]

SMOKE_HOOK_MODES = {
    "source_plain": "capture_only", "source_hooked": "donor_replace",
    "target_plain": "capture_only", "target_pasteback": "donor_replace",
}


def _science_hook_mode(family: str, word: str) -> str:
    if family in ("target_1", "target_2", "target_3"):
        return "capture_only" if word == "epsilon" else "plain"
    if family == "source":
        return "self_replace"
    return "donor_replace"


def _science_recipe(family: str, entity: dict) -> str | None:
    if family in ("target_1", "target_2", "target_3"):
        return None
    if family == "source":
        return _recipe_hash({"type": "self_identity"})
    if family == "pasteback":
        return _recipe_hash({"type": "donor_replace", "donor_row": entity["target_rows"][0]})
    if family == "native":
        return _recipe_hash({"type": "donor_replace", "donor_row": entity["target_rows"][0]})
    if family == "centroid":
        return _recipe_hash({"type": "centroid", "donor_rows": entity["correct_centroid_donor_rows"]})
    if family == "wrong":
        return _recipe_hash({"type": "centroid", "donor_rows": entity["wrong_centroid_donor_rows"]})
    return None


def build_call_table(cfg: dict, run_phase: str = "science") -> list[dict]:
    entities = cfg["entities"]
    if run_phase == "smoke":
        smoke_entities = cfg["smoke"]["entities"]
        smoke_words = cfg["smoke"]["words"]
        calls = []
        for replay in ["A", "B"]:
            for eid in smoke_entities:
                for wname in smoke_words:
                    for fam in SMOKE_FAMILIES:
                        recipe = _recipe_hash({"type": "self_identity"}) if fam == "source_hooked" else None
                        calls.append({
                            "run_phase": "smoke", "replay": replay, "entity_id": eid,
                            "family": fam, "word": wname,
                            "hook_mode": SMOKE_HOOK_MODES[fam],
                            "recipe_hash": recipe,
                        })
        assert len(calls) == 32, f"Smoke call count {len(calls)} != 32"
        return calls

    calls = []
    for replay in ["A", "B"]:
        for eid in range(24):
            e = entities[eid]
            for fam in FAMILIES:
                for wname in WORDS:
                    calls.append({
                        "run_phase": "science", "replay": replay, "entity_id": eid,
                        "family": fam, "word": wname,
                        "hook_mode": _science_hook_mode(fam, wname),
                        "recipe_hash": _science_recipe(fam, e),
                    })
    assert len(calls) == 2688, f"Science call count {len(calls)} != 2688"

    phase_d = [c for c in calls if c["family"] in ("target_1", "target_2", "target_3") and c["word"] == "epsilon"]
    phase_e = [c for c in calls if c not in phase_d]
    assert len(phase_d) == 144, f"Phase D count {len(phase_d)} != 144 (72 per replay)"
    assert len(phase_e) == 2544, f"Phase E count {len(phase_e)} != 2544 (1272 per replay)"
    return calls


def phase_d_calls(calls: list[dict], replay: str) -> list[dict]:
    d = [c for c in calls if c["replay"] == replay and c["family"] in ("target_1", "target_2", "target_3") and c["word"] == "epsilon"]
    if replay == "A":
        d.sort(key=lambda c: (c["entity_id"], FAMILIES.index(c["family"])))
    else:
        d.sort(key=lambda c: (c["entity_id"], FAMILIES.index(c["family"])), reverse=True)
    return d


def phase_e_calls(calls: list[dict], replay: str) -> list[dict]:
    d_set = {(c["replay"], c["entity_id"], c["family"], c["word"]) for c in calls
             if c["family"] in ("target_1", "target_2", "target_3") and c["word"] == "epsilon"}
    e = [c for c in calls if c["replay"] == replay and (c["replay"], c["entity_id"], c["family"], c["word"]) not in d_set]
    key = lambda c: (c["entity_id"], FAMILIES.index(c["family"]), WORDS.index(c["word"]))
    if replay == "A":
        e.sort(key=key)
    else:
        e.sort(key=key, reverse=True)
    return e


def call_identity(c: dict) -> str:
    return (f"{c['run_phase']}|{c['replay']}|{c['entity_id']}|{c['family']}|{c['word']}"
            f"|{c['hook_mode']}|{c.get('recipe_hash') or ''}")

# --------------- hook state machine ---------------

@dataclass
class HookState:
    mode: str = "inactive"
    position: int = -1
    payload: torch.Tensor | None = None
    captured: torch.Tensor | None = None
    captures: int = 0
    writes: int = 0

# --------------- model wrapper ---------------

class BridgeModel:
    def __init__(self, cfg: dict, rows_data: list[dict]):
        self.cfg = cfg
        self.rows_data = rows_data
        self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"])
        assert self.sp.revision == cfg["revision"]
        self.model = self.sp.model
        self.tok = self.sp.tok
        torch.set_grad_enabled(False)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layer_module = self.model.model.layers[cfg["layer"]]
        self._state = HookState()
        self._hook_handle = self.layer_module.register_forward_hook(self._hook)

        self.ids = lambda t: self.tok.encode(t, add_special_tokens=False)

        self.numeral_ids = []
        for d in cfg["numerals"]:
            tids = self.ids(d)
            assert len(tids) == 1, f"Numeral '{d}' is not a single token: {tids}"
            self.numeral_ids.append(tids[0])

        self._build_entity_queries()

    def _hook(self, module, inp, out):
        if self._state.mode == "inactive":
            return out
        h = out[0] if isinstance(out, tuple) else out
        assert h.ndim == 3 and h.shape[0] == 1
        p = self._state.position
        assert 0 <= p < h.shape[1]
        original = h[0, p, :].detach().clone()
        self._state.captured = original
        self._state.captures += 1
        if self._state.mode == "capture_only":
            assert torch.isfinite(original).all(), "Non-finite vector at capture site"
            return out
        payload = original if self._state.mode == "self_replace" else self._state.payload
        assert payload.shape == original.shape
        assert torch.isfinite(payload).all()
        patched = h.clone()
        patched[0, p, :] = payload.to(device=h.device, dtype=h.dtype)
        self._state.writes += 1
        return (patched,) + tuple(out[1:]) if isinstance(out, tuple) else patched

    def _build_entity_queries(self):
        queries = self.cfg["queries"]
        self.entity_queries = {}
        for e in self.cfg["entities"]:
            idx = e["query_idx"]
            q_str = queries[idx - 1].format(entity=e["name"])
            q_ids = self.ids(q_str)
            p_str = self.tok.decode([q_ids[0]])
            p_ids = [q_ids[0]]
            self.entity_queries[e["id"]] = {
                "Q_str": q_str, "Q_ids": q_ids,
                "P_str": p_str, "P_ids": p_ids
            }

    def get_row_tokens(self, row_idx: int) -> list[int]:
        row = self.rows_data[row_idx]
        prompt_text = row["prompt"]
        tids = self.ids(prompt_text)
        stored = row.get("ids")
        if stored is not None and tids != stored:
            return None
        return tids

    def get_site_pos(self, row_idx: int) -> int:
        row = self.rows_data[row_idx]
        span = row["span"]
        return span[1] - 1

    def build_input_ids(self, row_idx: int, word: str, entity_id: int) -> list[int]:
        base = self.get_row_tokens(row_idx)
        if base is None:
            return None
        if word == "epsilon":
            return base
        eq = self.entity_queries[entity_id]
        word_def = self.cfg["words"][word]
        actions = word_def["actions"]
        # D1 rightmost-first: display a_Qa_P = ["Q","P"] executes P first then Q.
        # In autoregressive LM, execution order = left-to-right token order.
        # So tokens are appended in REVERSED action order: P_ids then Q_ids.
        exec_order = list(reversed(actions))
        token_ids = list(base)
        for action in exec_order:
            if action == "P":
                token_ids.extend(eq["P_ids"])
            elif action == "Q":
                token_ids.extend(eq["Q_ids"])
        return token_ids

    def _run_forward(self, input_ids: list[int]) -> object:
        ids_t = torch.tensor([input_ids])
        L = len(input_ids)
        attn = torch.ones(1, L, dtype=torch.long)
        pos = torch.arange(L).unsqueeze(0)
        return self.model(input_ids=ids_t, attention_mask=attn, position_ids=pos,
                          past_key_values=None, use_cache=False)

    def _extract_law(self, out) -> np.ndarray:
        logits = out.logits[0, -1].float()
        law = torch.log_softmax(logits, dim=-1).cpu().numpy()
        if not np.all(np.isfinite(law)):
            raise ValueError("Non-finite values in extracted law")
        return law

    @torch.no_grad()
    def forward_plain(self, input_ids: list[int]) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        self._state = HookState(mode="inactive")
        try:
            out = self._run_forward(input_ids)
            assert self._state.captures == 0 and self._state.writes == 0
        finally:
            self._state = HookState()
        return self._extract_law(out), {"seconds": time.time() - t0, "type": "plain"}

    @torch.no_grad()
    def forward_capture(self, input_ids: list[int], site_pos: int) -> tuple[np.ndarray, dict, torch.Tensor]:
        t0 = time.time()
        self._state = HookState(mode="capture_only", position=site_pos)
        try:
            out = self._run_forward(input_ids)
            captured = self._state.captured
            assert self._state.captures == 1 and self._state.writes == 0
            assert torch.isfinite(captured).all(), "Non-finite captured vector"
        finally:
            self._state = HookState()
        return self._extract_law(out), {"seconds": time.time() - t0, "type": "capture"}, captured

    @torch.no_grad()
    def forward_self_replace(self, input_ids: list[int], site_pos: int) -> tuple[np.ndarray, dict, torch.Tensor]:
        t0 = time.time()
        self._state = HookState(mode="self_replace", position=site_pos)
        try:
            out = self._run_forward(input_ids)
            captured = self._state.captured
            assert self._state.captures == 1 and self._state.writes == 1
            assert torch.isfinite(captured).all(), "Non-finite captured vector"
        finally:
            self._state = HookState()
        return self._extract_law(out), {"seconds": time.time() - t0, "type": "self_replace"}, captured

    @torch.no_grad()
    def forward_hooked(self, input_ids: list[int], replace_vec: torch.Tensor, site_pos: int) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        self._state = HookState(mode="donor_replace", position=site_pos, payload=replace_vec)
        try:
            out = self._run_forward(input_ids)
            assert self._state.captures == 1 and self._state.writes == 1
            assert torch.isfinite(self._state.captured).all(), "Non-finite captured vector in hooked forward"
        finally:
            self._state = HookState()
        return self._extract_law(out), {"seconds": time.time() - t0, "type": "hooked"}

# --------------- smoke stage ---------------

def run_smoke(cfg: dict, bm: BridgeModel, rows_data: list[dict], out_dir: str, log_fn, smoke_call_hash: str = "") -> dict:
    log_fn("=== SMOKE STAGE ===")
    smoke_entities = cfg["smoke"]["entities"]
    smoke_words = cfg["smoke"]["words"]
    consts = cfg["constants"]
    timings_plain = []
    timings_hooked = []
    laws = {}

    for replay in ["A", "B"]:
        log_fn(f"Smoke replay {replay}")
        for eid in smoke_entities:
            e = cfg["entities"][eid]
            src_row = e["source_row"]
            tgt_row = e["target_rows"][0]
            src_site_pos = bm.get_site_pos(src_row)
            tgt_site_pos = bm.get_site_pos(tgt_row)

            for wname in smoke_words:
                src_ids = bm.build_input_ids(src_row, wname, eid)
                tgt_ids = bm.build_input_ids(tgt_row, wname, eid)
                if src_ids is None or tgt_ids is None:
                    return {"status": "INVALID — ROW/TOKENIZATION MANIFEST"}

                # source_plain (capture_only: captures src vec, output unchanged)
                lp, t, src_vec = bm.forward_capture(src_ids, src_site_pos)
                timings_plain.append(t["seconds"])
                laws[(replay, eid, "source_plain", wname)] = lp

                # source_hooked (donor_replace with captured src vec = identity)
                lp, t = bm.forward_hooked(src_ids, src_vec, src_site_pos)
                timings_hooked.append(t["seconds"])
                laws[(replay, eid, "source_hooked", wname)] = lp

                # target_plain (capture_only: captures tgt vec, output unchanged)
                lp, t, tgt_vec = bm.forward_capture(tgt_ids, tgt_site_pos)
                timings_plain.append(t["seconds"])
                laws[(replay, eid, "target_plain", wname)] = lp

                # target_pasteback (donor_replace with captured tgt vec = identity)
                lp, t = bm.forward_hooked(tgt_ids, tgt_vec, tgt_site_pos)
                timings_hooked.append(t["seconds"])
                laws[(replay, eid, "target_pasteback", wname)] = lp

    # Smoke completeness gate
    if len(laws) != 32:
        return {"status": "INVALID — INCOMPLETE SMOKE", "collected": len(laws), "expected": 32}

    # Compute eta_smoke: max same-tuple A/B discrepancy across both channels
    eta_smoke = 0.0
    replay_details = {}
    for key_a in laws:
        if key_a[0] != "A":
            continue
        key_b = ("B",) + key_a[1:]
        assert key_b in laws, f"Missing B-replay partner for {key_a}"
        for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
            d = sqrt_js(get_ch(laws[key_a]), get_ch(laws[key_b]))
            tag = f"{key_a[1]}|{key_a[2]}|{key_a[3]}|{ch_name}"
            replay_details[tag] = d
            if d > eta_smoke:
                eta_smoke = d

    log_fn(f"eta_smoke = {eta_smoke:.8f}")
    if eta_smoke > consts["replay_invalidity_ceiling"]:
        return {"status": "INVALID — NUMERICAL REPLAY", "eta_smoke": eta_smoke}

    eps_smoke = max(consts["epsilon_0"], 2.0 * eta_smoke)
    log_fn(f"epsilon_smoke = {eps_smoke:.8f}")

    # Fixture 1: plain-target vs same-carrier paste-back (preregistered)
    # Fixture 2: plain-source vs unchanged-hook source (preregistered)
    fixture_results = {}
    fixture_ok = True
    for replay in ["A", "B"]:
        for eid in smoke_entities:
            for wname in smoke_words:
                for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
                    d_pb = sqrt_js(get_ch(laws[(replay, eid, "target_plain", wname)]),
                                   get_ch(laws[(replay, eid, "target_pasteback", wname)]))
                    tag_pb = f"pasteback|{replay}|{eid}|{wname}|{ch_name}"
                    fixture_results[tag_pb] = d_pb
                    if d_pb > eps_smoke:
                        log_fn(f"FIXTURE FAIL {tag_pb}: {d_pb:.8f} > {eps_smoke:.8f}")
                        fixture_ok = False

                    d_sh = sqrt_js(get_ch(laws[(replay, eid, "source_plain", wname)]),
                                   get_ch(laws[(replay, eid, "source_hooked", wname)]))
                    tag_sh = f"source_hook|{replay}|{eid}|{wname}|{ch_name}"
                    fixture_results[tag_sh] = d_sh
                    if d_sh > eps_smoke:
                        log_fn(f"FIXTURE FAIL {tag_sh}: {d_sh:.8f} > {eps_smoke:.8f}")
                        fixture_ok = False

    if not fixture_ok:
        return {"status": "INVALID — SITE CARRIER/HOOK", "eta_smoke": eta_smoke,
                "epsilon_smoke": eps_smoke, "fixtures": fixture_results}

    # QP token-order fixture (exact)
    for eid in smoke_entities:
        eq = bm.entity_queries[eid]
        expected = eq["P_ids"] + eq["Q_ids"]
        base = bm.get_row_tokens(cfg["entities"][eid]["source_row"])
        actual_input = bm.build_input_ids(cfg["entities"][eid]["source_row"], "a_Qa_P", eid)
        actual_suffix = actual_input[len(base):]
        if actual_suffix != expected:
            return {"status": f"INVALID — QP TOKEN ORDER: entity {eid}"}

    # Timings and forecast
    s_plain = float(np.mean(timings_plain))
    s_hooked = float(np.mean(timings_hooked))
    s_smoke = max(s_plain, s_hooked)
    F_CPU = 1.5 * s_smoke * 2688 / 60.0
    H_CPU = 5 * math.ceil(F_CPU / 5)

    log_fn(f"Timings: plain={s_plain:.3f}s hooked={s_hooked:.3f}s s_smoke={s_smoke:.3f}s")
    log_fn(f"Forecast: F_CPU={F_CPU:.1f} min, H_CPU={H_CPU} min, ceiling={consts['cpu_forecast_ceiling_minutes']} min")
    forecast_ok = F_CPU <= consts["cpu_forecast_ceiling_minutes"] and H_CPU <= consts["cpu_forecast_ceiling_minutes"]

    return {
        "status": "SMOKE_VALID" if forecast_ok else "SMOKE_VALID_FORECAST_EXCEEDS_CEILING",
        "eta_smoke": eta_smoke, "epsilon_smoke": eps_smoke,
        "s_plain": s_plain, "s_hooked": s_hooked, "s_smoke": s_smoke,
        "F_CPU": F_CPU, "H_CPU": H_CPU,
        "n_calls": len(laws), "forecast_ok": forecast_ok,
        "fixtures": fixture_results, "replay_discrepancies": replay_details,
        "call_table_hash": smoke_call_hash,
    }

# --------------- tokenizer + resample helpers ---------------

def _tokenizer_file_hashes(model_id: str, revision: str) -> dict[str, str]:
    from huggingface_hub import snapshot_download
    snap = snapshot_download(model_id, revision=revision, local_files_only=True)
    tok_files = ["tokenizer.json", "tokenizer_config.json"]
    hashes = {}
    for f in tok_files:
        fp = os.path.join(snap, f)
        if os.path.exists(fp):
            hashes[f] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    return hashes


def _build_resample_index(seed: int, n_resamples: int, n_entities: int) -> tuple[np.ndarray, str]:
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, n_entities, size=(n_resamples, n_entities), dtype=np.int64)
    idx_hash = hashlib.sha256(idx.tobytes()).hexdigest()
    return idx, idx_hash


# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["smoke", "science"])
    ap.add_argument("--lock-row", default=None)
    a = ap.parse_args()
    T0 = time.time()

    cfg = json.load(open(a.config, encoding="utf-8"))
    out_dir = cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, f"{a.stage}.log"), "w")
    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    # Stage 1: verify authorities
    log("Stage 1: Verifying authority hashes...")
    shas = verify_authorities(cfg)
    if shas is None:
        log("INVALID — AUTHORITY HASH MISMATCH")
        return

    # Stage 1: load rows and build manifest
    rows_data = json.load(open(cfg["authorities"]["run_rows"]["path"]))
    manifest = build_row_manifest(cfg, rows_data)
    if isinstance(manifest, str):
        log(manifest)
        return
    log(f"Row manifest: {manifest['count']} rows, hash={manifest['hash'][:16]}...")

    # Stage 2: enumerate BOTH call tables (smoke and science)
    log("Stage 2: Enumerating call tables...")
    smoke_calls = build_call_table(cfg, "smoke")
    science_calls = build_call_table(cfg, "science")
    log(f"Smoke calls: {len(smoke_calls)}, Science calls: {len(science_calls)}")

    smoke_call_ids = [call_identity(c) for c in smoke_calls]
    smoke_call_hash = hashlib.sha256(json.dumps(smoke_call_ids).encode()).hexdigest()
    science_call_ids = [call_identity(c) for c in science_calls]
    science_call_hash = hashlib.sha256(json.dumps(science_call_ids).encode()).hexdigest()
    log(f"Smoke call table hash: {smoke_call_hash[:16]}...")
    log(f"Science call table hash: {science_call_hash[:16]}...")

    runner_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    config_sha = hashlib.sha256(open(a.config, "rb").read()).hexdigest()

    # Pre-commit bootstrap resample index
    resample_idx, resample_hash = _build_resample_index(
        cfg["constants"]["bootstrap_seed"], cfg["constants"]["resamples_B"], 24)
    log(f"Resample index: shape={resample_idx.shape}, hash={resample_hash[:16]}...")

    # Stage 3: load model
    log("Stage 3: Loading model...")
    bm = BridgeModel(cfg, rows_data)
    log(f"Model loaded: {bm.sp.model_id}, revision={bm.sp.revision}")

    # Stage 4: retokenize and verify (strengthened row/span/donor validation)
    log("Stage 4: Retokenizing and verifying...")
    for e in cfg["entities"]:
        for row_idx in [e["source_row"]] + e["target_rows"]:
            row = rows_data[row_idx]
            tids = bm.get_row_tokens(row_idx)
            if tids is None:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: row {row_idx}")
                return
            stored_ids = row.get("ids")
            if stored_ids is not None and tids != stored_ids:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: retokenization mismatch row {row_idx}")
                return
            stored_len = row.get("len")
            if stored_len is not None and stored_len != len(tids):
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: len mismatch row {row_idx}")
                return
            span = row.get("span")
            if span is None or len(span) != 2:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: missing/bad span for row {row_idx}")
                return
            if span[0] < 0 or span[1] > len(tids) or span[0] >= span[1]:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: span out of range for row {row_idx}")
                return
            site_pos = span[1] - 1
            if site_pos < 0 or site_pos >= len(tids):
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: site pos {site_pos} out of range for row {row_idx}")
                return
            span_ids = tids[span[0]:span[1]]
            span_text = row.get("span_text")
            if span_text is not None:
                decoded = bm.tok.decode(span_ids)
                if decoded != span_text:
                    log(f"INVALID — ROW/TOKENIZATION MANIFEST: span_text mismatch row {row_idx}")
                    return
        # Verify donor membership
        all_phase_d_target_rows = set()
        for ent in cfg["entities"]:
            for tr in ent["target_rows"]:
                all_phase_d_target_rows.add(tr)
        for dr in e["correct_centroid_donor_rows"]:
            if dr not in all_phase_d_target_rows:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: correct donor {dr} not a Phase D target row")
                return
        for dr in e["wrong_centroid_donor_rows"]:
            if dr not in all_phase_d_target_rows:
                log(f"INVALID — ROW/TOKENIZATION MANIFEST: wrong donor {dr} not a Phase D target row")
                return
    # Verify numeral tokens
    for i, d in enumerate(cfg["numerals"]):
        tids = bm.ids(d)
        if len(tids) != 1 or tids[0] != bm.numeral_ids[i]:
            log(f"INVALID — NUMERAL TOKEN: '{d}' -> {tids}")
            return
    log("Retokenization, site positions, spans, and donors verified")

    # Materialize token IDs into call tables
    log("Materializing token IDs into call tables...")
    for call_list in [smoke_calls, science_calls]:
        for c in call_list:
            eid = c["entity_id"]
            e = cfg["entities"][eid]
            fam = c["family"]
            wname = c["word"]
            if fam in SMOKE_FAMILIES:
                if fam in ("source_plain", "source_hooked"):
                    row_idx = e["source_row"]
                else:
                    row_idx = e["target_rows"][0]
            elif fam in ("target_1", "target_2", "target_3"):
                fi = FAMILIES.index(fam)
                row_idx = e["target_rows"][fi]
            elif fam == "pasteback":
                row_idx = e["target_rows"][0]
            else:
                row_idx = e["source_row"]
            input_ids = bm.build_input_ids(row_idx, wname, eid)
            assert input_ids is not None, f"Failed to build input_ids for {call_identity(c)}"
            c["input_ids"] = input_ids
            c["row_idx"] = row_idx
    # Recompute call table hashes with materialized token IDs
    smoke_call_ids = [call_identity(c) for c in smoke_calls]
    smoke_call_hash = hashlib.sha256(json.dumps(smoke_call_ids).encode()).hexdigest()
    science_call_ids = [call_identity(c) for c in science_calls]
    science_call_hash = hashlib.sha256(json.dumps(science_call_ids).encode()).hexdigest()
    log(f"Materialized smoke call hash: {smoke_call_hash[:16]}...")
    log(f"Materialized science call hash: {science_call_hash[:16]}...")

    # Build expanded prospective science manifest
    import platform, torch as _torch
    import transformers, tokenizers as _tokenizers_mod
    tok_file_hashes = _tokenizer_file_hashes(cfg["model_id"], cfg["revision"])
    full_manifest = {
        "row_manifest": manifest,
        "runner_sha256": runner_sha,
        "config_sha256": config_sha,
        "smoke_call_table_hash": smoke_call_hash,
        "science_call_table_hash": science_call_hash,
        "science_call_count": 2688,
        "smoke_call_count": 32,
        "model_id": bm.sp.model_id,
        "model_revision": bm.sp.revision,
        "tokenizer_file_hashes": tok_file_hashes,
        "vocab_size": bm.model.config.vocab_size,
        "device": str(next(bm.model.parameters()).device),
        "dtype": str(next(bm.model.parameters()).dtype),
        "batch_size": 1,
        "layer": cfg["layer"],
        "numeral_ids": bm.numeral_ids,
        "constants": cfg["constants"],
        "authorities": {k: v["sha256"] for k, v in cfg["authorities"].items()},
        "resample_index_hash": resample_hash,
        "resample_shape": list(resample_idx.shape),
        "resample_seed": cfg["constants"]["bootstrap_seed"],
        "canonical_science_argv": f"python experiments/run_native_bridge.py --config {a.config} --stage science --lock-row {cfg['results_dir']}/lock_row.json",
        "status_tree_version": cfg.get("status_tree", {}).get("version", "round41_audit47"),
        "stop_rule": "no_post_output_repair_or_rerun",
        "python_version": platform.python_version(),
        "torch_version": _torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": _tokenizers_mod.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    manifest_hash = hashlib.sha256(json.dumps(full_manifest, sort_keys=True).encode()).hexdigest()
    full_manifest["manifest_hash"] = manifest_hash
    manifest_path = os.path.join(out_dir, "manifest.json")
    log(f"Manifest hash: {manifest_hash[:16]}...")

    # QP token-order fixture: D1 rightmost-first means a_Qa_P appends P_ids || Q_ids
    for e in cfg["entities"]:
        eq = bm.entity_queries[e["id"]]
        expected_suffix = eq["P_ids"] + eq["Q_ids"]
        qp_input = bm.build_input_ids(e["source_row"], "a_Qa_P", e["id"])
        base = bm.get_row_tokens(e["source_row"])
        actual_suffix = qp_input[len(base):]
        if actual_suffix != expected_suffix:
            log(f"INVALID — QP TOKEN ORDER: entity {e['id']}, expected {expected_suffix}, got {actual_suffix}")
            return

    if a.stage == "smoke":
        result = run_smoke(cfg, bm, rows_data, out_dir, log, smoke_call_hash=smoke_call_hash)
        result["runner_sha256"] = runner_sha
        result["config_sha256"] = config_sha
        result["manifest_hash"] = manifest_hash
        result["seconds"] = time.time() - T0
        smoke_json = json.dumps(result, sort_keys=True, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o) if isinstance(o, (np.floating,)) else o)
        result["smoke_artifact_hash"] = hashlib.sha256(smoke_json.encode()).hexdigest()
        # Write smoke commitment into manifest and save
        full_manifest["smoke_commitment"] = {
            "status": result.get("status"),
            "eta_smoke": result.get("eta_smoke"),
            "epsilon_smoke": result.get("epsilon_smoke"),
            "s_plain": result.get("s_plain"),
            "s_hooked": result.get("s_hooked"),
            "s_smoke": result.get("s_smoke"),
            "F_CPU": result.get("F_CPU"),
            "H_CPU": result.get("H_CPU"),
            "forecast_ok": result.get("forecast_ok"),
            "smoke_artifact_hash": result["smoke_artifact_hash"],
        }
        # Recompute manifest hash with smoke commitment included
        manifest_hash = hashlib.sha256(json.dumps(full_manifest, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o) if isinstance(o, (np.floating,)) else o).encode()).hexdigest()
        full_manifest["manifest_hash"] = manifest_hash
        result["manifest_hash"] = manifest_hash
        json.dump(full_manifest, open(manifest_path, "w"), indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o) if isinstance(o, (np.floating,)) else o)
        log(f"Manifest written with smoke commitment: hash={manifest_hash[:16]}...")
        # Save resample index
        np.save(os.path.join(out_dir, "resample_index.npy"), resample_idx)
        json.dump(result, open(os.path.join(out_dir, "smoke_result.json"), "w"), indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o) if isinstance(o, (np.floating,)) else o)
        log(f"Smoke complete: {result['status']} ({time.time()-T0:.0f}s)")
        if not result.get("forecast_ok"):
            f_cpu = result.get("F_CPU")
            h_cpu = result.get("H_CPU")
            if f_cpu is not None and h_cpu is not None:
                log(f"ABORT: F_CPU={f_cpu:.1f} or H_CPU={h_cpu} exceeds {cfg['constants']['cpu_forecast_ceiling_minutes']} min ceiling")
            else:
                log(f"ABORT: smoke status={result['status']} (no forecast available)")
        return

    # Stage 7: verify retained manifest and lock row
    log("Stage 7: Verifying manifest and lock row...")
    # Load and verify the retained manifest (written during smoke)
    if not os.path.exists(manifest_path):
        log("REFUSED — manifest.json missing; run smoke first")
        return
    retained_manifest = json.load(open(manifest_path))
    if retained_manifest.get("manifest_hash") != manifest_hash:
        # The manifest was written during smoke with smoke_commitment; recompute with it
        if "smoke_commitment" in retained_manifest:
            check_manifest = dict(full_manifest)
            check_manifest["smoke_commitment"] = retained_manifest["smoke_commitment"]
            check_hash = hashlib.sha256(json.dumps(check_manifest, sort_keys=True,
                      default=lambda o: o.item() if hasattr(o, "item") else float(o) if isinstance(o, (np.floating,)) else o).encode()).hexdigest()
            if retained_manifest.get("manifest_hash") != check_hash:
                log(f"REFUSED — manifest hash mismatch: retained={retained_manifest.get('manifest_hash')}, recomputed={check_hash}")
                return
            manifest_hash = check_hash
        else:
            log(f"REFUSED — manifest hash mismatch and no smoke_commitment found")
            return
    log(f"Manifest verified: hash={manifest_hash[:16]}...")

    # Verify resample index
    resample_path = os.path.join(out_dir, "resample_index.npy")
    if not os.path.exists(resample_path):
        log("REFUSED — resample_index.npy missing; run smoke first")
        return
    stored_resample = np.load(resample_path)
    stored_resample_hash = hashlib.sha256(stored_resample.tobytes()).hexdigest()
    if stored_resample_hash != resample_hash:
        log(f"REFUSED — resample index hash mismatch: stored={stored_resample_hash}, computed={resample_hash}")
        return
    resample_idx = stored_resample
    log(f"Resample index verified: hash={resample_hash[:16]}...")

    # Verify lock row
    if a.lock_row is None:
        log("REFUSED — no --lock-row provided; scientific mode requires a valid pre-science lock row")
        return
    lock = json.load(open(a.lock_row))
    required_lock_fields = ["runner_sha256", "config_sha256", "call_count", "manifest_hash",
                            "call_table_hash", "smoke_status", "smoke_artifact_hash",
                            "H_CPU", "eta_smoke", "epsilon_smoke", "s_smoke", "F_CPU",
                            "constants", "stop_rule"]
    missing = [f for f in required_lock_fields if f not in lock]
    if missing:
        log(f"REFUSED — lock row missing fields: {missing}")
        return
    if lock["runner_sha256"] != runner_sha:
        log(f"REFUSED — runner hash mismatch: lock={lock['runner_sha256']}, actual={runner_sha}")
        return
    if lock["config_sha256"] != config_sha:
        log(f"REFUSED — config hash mismatch: lock={lock['config_sha256']}, actual={config_sha}")
        return
    if lock["call_count"] != 2688:
        log(f"REFUSED — lock row call count {lock['call_count']} != 2688")
        return
    if lock["manifest_hash"] != manifest_hash:
        log(f"REFUSED — manifest hash mismatch: lock={lock['manifest_hash']}, actual={manifest_hash}")
        return
    if lock["call_table_hash"] != science_call_hash:
        log(f"REFUSED — call table hash mismatch: lock={lock['call_table_hash']}, actual={science_call_hash}")
        return
    if lock["smoke_status"] != "SMOKE_VALID":
        log(f"REFUSED — lock row smoke was not SMOKE_VALID: {lock['smoke_status']}")
        return
    # Verify smoke artifact hash against retained smoke result
    smoke_result_path = os.path.join(out_dir, "smoke_result.json")
    if os.path.exists(smoke_result_path):
        stored_smoke = json.load(open(smoke_result_path))
        if stored_smoke.get("smoke_artifact_hash") != lock["smoke_artifact_hash"]:
            log(f"REFUSED — smoke artifact hash mismatch")
            return
    # Verify forecast is within ceiling
    if lock["F_CPU"] > cfg["constants"]["cpu_forecast_ceiling_minutes"]:
        log(f"REFUSED — F_CPU {lock['F_CPU']} exceeds ceiling {cfg['constants']['cpu_forecast_ceiling_minutes']}")
        return
    if lock["H_CPU"] > cfg["constants"]["cpu_forecast_ceiling_minutes"]:
        log(f"REFUSED — H_CPU {lock['H_CPU']} exceeds ceiling {cfg['constants']['cpu_forecast_ceiling_minutes']}")
        return
    if lock["stop_rule"] != "no_post_output_repair_or_rerun":
        log(f"REFUSED — stop rule mismatch: {lock['stop_rule']}")
        return
    log(f"Lock row verified: runner={runner_sha[:16]}... config={config_sha[:16]}...")

    consts = cfg["constants"]
    calls = science_calls
    # Hard-wall clock starts HERE, after all verification, not at T0
    science_t0 = time.monotonic()
    deadline = science_t0 + lock["H_CPU"] * 60

    # Open checkpoint journal
    ckpt_path = os.path.join(out_dir, "call_checkpoints.jsonl")
    ckpt_f = open(ckpt_path, "w")

    # Stage 8: execute Phase D + Phase E per replay
    log("Stage 8: Scientific execution...")
    laws = {}         # (replay, entity_id, family, word) -> log_probs (full vocab)
    donors = {}       # (replay, target_row_id) -> site_vec tensor
    timings = []
    call_ordinal = 0

    def _checkpoint_call(c, lp, t_info, payload_hash=None, donor_hash=None):
        nonlocal call_ordinal
        law_hash = hashlib.sha256(lp.tobytes()).hexdigest()
        entry = {
            "ordinal": call_ordinal,
            "identity": call_identity(c),
            "hook_mode": c["hook_mode"],
            "recipe_hash": c.get("recipe_hash"),
            "law_hash": law_hash,
            "seconds": t_info["seconds"],
        }
        if payload_hash is not None:
            entry["payload_hash"] = payload_hash
        if donor_hash is not None:
            entry["donor_hash"] = donor_hash
        ckpt_f.write(json.dumps(entry) + "\n")
        ckpt_f.flush()
        call_ordinal += 1

    def _check_wall():
        if time.monotonic() > deadline:
            ckpt_f.close()
            result = {"status": "INCOMPLETE — CPU HARD WALL", "completed_calls": len(laws),
                      "science_seconds": time.monotonic() - science_t0,
                      "total_seconds": time.time() - T0}
            json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
            log(f"HARD WALL at call {len(laws)} ({time.monotonic()-science_t0:.0f}s science)")
            return True
        return False

    for replay in ["A", "B"]:
        log(f"Replay {replay}: Phase D (72 target-epsilon calls)...")
        d_calls = phase_d_calls(calls, replay)
        for ci, c in enumerate(d_calls):
            if _check_wall():
                return
            e = cfg["entities"][c["entity_id"]]
            tgt_family_idx = FAMILIES.index(c["family"])
            tgt_row = e["target_rows"][tgt_family_idx]
            tgt_site_pos = bm.get_site_pos(tgt_row)
            input_ids = bm.build_input_ids(tgt_row, c["word"], c["entity_id"])
            lp, t, site_vec = bm.forward_capture(input_ids, tgt_site_pos)
            timings.append(t["seconds"])
            laws[(replay, c["entity_id"], c["family"], c["word"])] = lp
            donors[(replay, tgt_row)] = site_vec
            donor_h = hashlib.sha256(site_vec.numpy().tobytes()).hexdigest()
            _checkpoint_call(c, lp, t, donor_hash=donor_h)
            if ci % 12 == 0:
                log(f"  D-{replay} call {ci}/{len(d_calls)} entity={c['entity_id']} ({time.monotonic()-science_t0:.0f}s)")

        # Entity checkpoint after Phase D
        ckpt_f.write(json.dumps({"marker": f"phase_d_complete_{replay}", "calls": len(laws)}) + "\n")
        ckpt_f.flush()

        log(f"Replay {replay}: Phase E (1272 remaining calls)...")
        e_calls = phase_e_calls(calls, replay)
        for ci, c in enumerate(e_calls):
            if _check_wall():
                return
            e = cfg["entities"][c["entity_id"]]
            src_row = e["source_row"]
            src_site_pos = bm.get_site_pos(src_row)
            payload_h = None

            if c["family"] in ("target_1", "target_2", "target_3"):
                tgt_family_idx = FAMILIES.index(c["family"])
                tgt_row = e["target_rows"][tgt_family_idx]
                input_ids = bm.build_input_ids(tgt_row, c["word"], c["entity_id"])
                lp, t = bm.forward_plain(input_ids)
                timings.append(t["seconds"])

            elif c["family"] == "pasteback":
                tgt_row = e["target_rows"][0]
                tgt_site_pos = bm.get_site_pos(tgt_row)
                input_ids = bm.build_input_ids(tgt_row, c["word"], c["entity_id"])
                paste_vec = donors.get((replay, tgt_row))
                if paste_vec is None:
                    log(f"INVALID — MISSING DONOR for pasteback replay={replay} row={tgt_row}")
                    return
                payload_h = hashlib.sha256(paste_vec.numpy().tobytes()).hexdigest()
                lp, t = bm.forward_hooked(input_ids, paste_vec, tgt_site_pos)
                timings.append(t["seconds"])

            elif c["family"] == "native":
                input_ids = bm.build_input_ids(src_row, c["word"], c["entity_id"])
                tgt_row = e["target_rows"][0]
                native_vec = donors.get((replay, tgt_row))
                if native_vec is None:
                    log(f"INVALID — MISSING DONOR for native replay={replay} row={tgt_row}")
                    return
                payload_h = hashlib.sha256(native_vec.numpy().tobytes()).hexdigest()
                lp, t = bm.forward_hooked(input_ids, native_vec, src_site_pos)
                timings.append(t["seconds"])

            elif c["family"] == "source":
                input_ids = bm.build_input_ids(src_row, c["word"], c["entity_id"])
                lp, t, _ = bm.forward_self_replace(input_ids, src_site_pos)
                timings.append(t["seconds"])

            elif c["family"] == "centroid":
                input_ids = bm.build_input_ids(src_row, c["word"], c["entity_id"])
                donor_rows = e["correct_centroid_donor_rows"]
                vecs = [donors[(replay, dr)] for dr in donor_rows if (replay, dr) in donors]
                if len(vecs) != len(donor_rows):
                    log(f"INVALID — MISSING CENTROID DONORS for entity {c['entity_id']} replay={replay}")
                    return
                centroid_vec = torch.stack(vecs).mean(dim=0)
                payload_h = hashlib.sha256(centroid_vec.numpy().tobytes()).hexdigest()
                lp, t = bm.forward_hooked(input_ids, centroid_vec, src_site_pos)
                timings.append(t["seconds"])

            elif c["family"] == "wrong":
                input_ids = bm.build_input_ids(src_row, c["word"], c["entity_id"])
                donor_rows = e["wrong_centroid_donor_rows"]
                vecs = [donors[(replay, dr)] for dr in donor_rows if (replay, dr) in donors]
                if len(vecs) != len(donor_rows):
                    log(f"INVALID — MISSING WRONG DONORS for entity {c['entity_id']} replay={replay}")
                    return
                wrong_vec = torch.stack(vecs).mean(dim=0)
                payload_h = hashlib.sha256(wrong_vec.numpy().tobytes()).hexdigest()
                lp, t = bm.forward_hooked(input_ids, wrong_vec, src_site_pos)
                timings.append(t["seconds"])

            else:
                log(f"INVALID — UNKNOWN FAMILY {c['family']}")
                return

            laws[(replay, c["entity_id"], c["family"], c["word"])] = lp
            _checkpoint_call(c, lp, t, payload_hash=payload_h)

            if ci % 100 == 0:
                log(f"  E-{replay} call {ci}/{len(e_calls)} ({time.monotonic()-science_t0:.0f}s)")

        replay_count = sum(1 for k in laws if k[0] == replay)
        ckpt_f.write(json.dumps({"marker": f"replay_complete_{replay}", "calls": replay_count}) + "\n")
        ckpt_f.flush()
        log(f"Replay {replay} complete: {replay_count} calls")

    ckpt_f.close()
    log(f"All {len(laws)} laws collected ({time.monotonic()-science_t0:.0f}s science, {time.time()-T0:.0f}s total)")

    # Completeness gate: every declared call must have a law
    if len(laws) != 2688:
        result = {"status": "INVALID — INCOMPLETE", "collected": len(laws), "expected": 2688,
                  "science_seconds": time.monotonic() - science_t0, "total_seconds": time.time() - T0}
        json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
        log(f"INVALID — INCOMPLETE: {len(laws)}/2688 laws collected")
        return
    expected_keys = {(c["replay"], c["entity_id"], c["family"], c["word"]) for c in calls}
    missing_keys = expected_keys - set(laws.keys())
    if missing_keys:
        result = {"status": "INVALID — MISSING KEYS", "missing_count": len(missing_keys),
                  "science_seconds": time.monotonic() - science_t0, "total_seconds": time.time() - T0}
        json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
        log(f"INVALID — MISSING KEYS: {len(missing_keys)} call keys not in laws")
        return

    # Stage 9: validate replay, derive metrics
    log("Stage 9: Validation and metric derivation...")

    # Compute eta: max same-tuple A/B discrepancy
    eta = 0.0
    for key_a in laws:
        if key_a[0] != "A":
            continue
        key_b = ("B",) + key_a[1:]
        if key_b not in laws:
            continue
        for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
            d = sqrt_js(get_ch(laws[key_a]), get_ch(laws[key_b]))
            if d > eta:
                eta = d

    log(f"eta = {eta:.8f}")
    if eta > consts["replay_invalidity_ceiling"]:
        result = {"status": "INVALID — NUMERICAL REPLAY", "eta": eta, "seconds": time.time() - T0}
        json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
        log(f"INVALID — NUMERICAL REPLAY: eta={eta}")
        return

    eps_B = max(consts["epsilon_0"], consts["k_B"] * eta)
    eps_E = max(consts["epsilon_0"], consts["k_E"] * eta)
    tau = eps_E + consts["delta"]
    log(f"eps_B={eps_B:.8f} eps_E={eps_E:.8f} tau={tau:.8f}")

    # Response totality check: every log-softmax must sum to ~1
    for key, lp in laws.items():
        total = float(np.exp(lp).sum())
        if abs(total - 1.0) > 1e-3:
            result = {"status": "INVALID — RESPONSE TOTALITY", "key": str(key), "total": total}
            json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
            log(f"INVALID — RESPONSE TOTALITY at {key}: sum={total}")
            return

    # Pasteback fixture on science calls
    for replay in ["A", "B"]:
        for eid in range(24):
            for wname in WORDS:
                tgt_law = laws[(replay, eid, "target_1", wname)]
                pb_law = laws[(replay, eid, "pasteback", wname)]
                for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
                    d = sqrt_js(get_ch(tgt_law), get_ch(pb_law))
                    if d > eps_B:
                        result = {"status": "INVALID — SITE CARRIER/HOOK",
                                  "replay": replay, "entity": eid, "word": wname,
                                  "channel": ch_name, "d": d, "eps_B": eps_B}
                        json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2)
                        log(f"INVALID — SITE CARRIER/HOOK: {replay}/{eid}/{wname}/{ch_name} d={d:.8f}")
                        return

    def compute_R_i(replay, eid, family):
        dists_per_target = []
        for tgt_fam in ["target_1", "target_2", "target_3"]:
            max_d = 0.0
            for wname in WORDS:
                edited = laws[(replay, eid, family, wname)]
                target = laws[(replay, eid, tgt_fam, wname)]
                for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
                    d = sqrt_js(get_ch(edited), get_ch(target))
                    if d > max_d:
                        max_d = d
            dists_per_target.append(max_d)
        return float(np.median(dists_per_target))

    def compute_V_i(replay, eid):
        pairs = [(0, 1), (0, 2), (1, 2)]
        dists = []
        for a, b in pairs:
            tgt_a_fam = f"target_{a+1}"
            tgt_b_fam = f"target_{b+1}"
            max_d = 0.0
            for wname in WORDS:
                la = laws[(replay, eid, tgt_a_fam, wname)]
                lb = laws[(replay, eid, tgt_b_fam, wname)]
                for ch_name, get_ch in [("c_full", lambda lp: lp), ("c_9", lambda lp: c9_from_full(lp, bm.numeral_ids))]:
                    d = sqrt_js(get_ch(la), get_ch(lb))
                    if d > max_d:
                        max_d = d
            dists.append(max_d)
        return float(np.median(dists))

    # Compute per-entity, per-replay estimands
    entity_data = {}
    for replay in ["A", "B"]:
        for eid in range(24):
            V_i = compute_V_i(replay, eid)
            R_nat = compute_R_i(replay, eid, "native")
            R_src = compute_R_i(replay, eid, "source")
            R_cent = compute_R_i(replay, eid, "centroid")
            R_wrong = compute_R_i(replay, eid, "wrong")
            E_nat = R_nat - V_i
            E_src = R_src - V_i
            E_cent = R_cent - V_i
            E_wrong = R_wrong - V_i
            entity_data[(replay, eid)] = {
                "V_i": V_i, "R_nat": R_nat, "R_src": R_src, "R_cent": R_cent, "R_wrong": R_wrong,
                "E_nat": E_nat, "E_src": E_src, "E_cent": E_cent, "E_wrong": E_wrong
            }

    # For each quantity, registered value = larger upper / smaller lower across replays
    def registered_bounds(metric_fn):
        vals_A = [metric_fn("A", eid) for eid in range(24)]
        vals_B = [metric_fn("B", eid) for eid in range(24)]
        mean_A = float(np.mean(vals_A))
        mean_B = float(np.mean(vals_B))
        return max(mean_A, mean_B), min(mean_A, mean_B), vals_A, vals_B

    # Theta_24 for each edit family
    Theta_nat_upper, Theta_nat_lower, E_nat_A, E_nat_B = registered_bounds(
        lambda r, i: entity_data[(r, i)]["E_nat"])
    Theta_src_upper, Theta_src_lower, E_src_A, E_src_B = registered_bounds(
        lambda r, i: entity_data[(r, i)]["E_src"])
    Theta_cent_upper, Theta_cent_lower, E_cent_A, E_cent_B = registered_bounds(
        lambda r, i: entity_data[(r, i)]["E_cent"])
    Theta_wrong_upper, Theta_wrong_lower, E_wrong_A, E_wrong_B = registered_bounds(
        lambda r, i: entity_data[(r, i)]["E_wrong"])

    # Delta_src and Delta_spec
    def delta_src_vals(replay, family):
        return [entity_data[(replay, i)][f"E_{family}"] - entity_data[(replay, i)]["E_src"]
                for i in range(24)]

    delta_src_nat_A = delta_src_vals("A", "nat")
    delta_src_nat_B = delta_src_vals("B", "nat")
    delta_src_cent_A = delta_src_vals("A", "cent")
    delta_src_cent_B = delta_src_vals("B", "cent")

    delta_spec_A = [entity_data[("A", i)]["E_cent"] - entity_data[("A", i)]["E_wrong"] for i in range(24)]
    delta_spec_B = [entity_data[("B", i)]["E_cent"] - entity_data[("B", i)]["E_wrong"] for i in range(24)]

    # Entity-cluster stability bounds (precommitted resample index)
    B = consts["resamples_B"]
    alpha = consts["stability_tail_alpha"]

    def stability_bounds(vals_per_entity):
        vals = np.array(vals_per_entity)
        boot = np.array([vals[resample_idx[b]].mean() for b in range(B)])
        return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))

    # For each registered quantity, take the registered bound (larger upper, smaller lower across replays)
    def registered_stability(vals_A, vals_B):
        L_A, U_A = stability_bounds(vals_A)
        L_B, U_B = stability_bounds(vals_B)
        return min(L_A, L_B), max(U_A, U_B)

    L_E_nat, U_E_nat = registered_stability(E_nat_A, E_nat_B)
    L_E_cent, U_E_cent = registered_stability(E_cent_A, E_cent_B)

    L_src_nat, U_src_nat = registered_stability(delta_src_nat_A, delta_src_nat_B)
    L_src_cent, U_src_cent = registered_stability(delta_src_cent_A, delta_src_cent_B)

    L_spec, U_spec = registered_stability(delta_spec_A, delta_spec_B)

    log(f"Theta_nat: [{Theta_nat_lower:.6f}, {Theta_nat_upper:.6f}]  bounds: L={L_E_nat:.6f} U={U_E_nat:.6f}")
    log(f"Delta_src(nat): L={L_src_nat:.6f} U={U_src_nat:.6f}")
    log(f"Theta_cent: [{Theta_cent_lower:.6f}, {Theta_cent_upper:.6f}]  bounds: L={L_E_cent:.6f} U={U_E_cent:.6f}")
    log(f"Delta_src(cent): L={L_src_cent:.6f} U={U_src_cent:.6f}")
    log(f"Delta_spec: L={L_spec:.6f} U={U_spec:.6f}")
    log(f"tau={tau:.6f}")

    # Stage 10: apply immutable status tree
    log("Stage 10: Status determination...")

    # Native paste status
    nat_pass = U_E_nat <= tau and U_src_nat <= -(eps_E + consts["delta_move"])
    nat_fail = L_E_nat > tau or L_src_nat > -(eps_E + consts["delta_move"])
    if nat_pass:
        nat_status = "NATIVE PASTE PASS"
    elif nat_fail:
        nat_status = "NATIVE PASTE FAIL"
    else:
        nat_status = "NATIVE PASTE INCONCLUSIVE"
    log(f"Native paste: {nat_status}")

    # Centroid status
    if nat_status != "NATIVE PASTE PASS":
        cent_status = "DIAGNOSTIC ONLY — PROXIMAL CONTROL NOT PASSED"
    else:
        cent_pass = (U_E_cent <= tau and
                     U_src_cent <= -(eps_E + consts["delta_move"]) and
                     U_spec <= -(eps_E + consts["delta_spec"]))
        cent_fail = (L_E_cent > tau or
                     L_src_cent > -(eps_E + consts["delta_move"]) or
                     L_spec > -(eps_E + consts["delta_spec"]))
        if cent_pass:
            cent_status = "CENTROID PASS"
        elif cent_fail:
            cent_status = "CENTROID REFUTATION"
        else:
            cent_status = "CENTROID INCONCLUSIVE"
    log(f"Centroid: {cent_status}")

    result = {
        "status": cent_status if nat_status == "NATIVE PASTE PASS" else nat_status,
        "native_paste_status": nat_status,
        "centroid_status": cent_status,
        "eta": eta, "eps_B": eps_B, "eps_E": eps_E, "tau": tau,
        "Theta_nat": {"lower": Theta_nat_lower, "upper": Theta_nat_upper,
                      "L_stability": L_E_nat, "U_stability": U_E_nat},
        "Theta_cent": {"lower": Theta_cent_lower, "upper": Theta_cent_upper,
                       "L_stability": L_E_cent, "U_stability": U_E_cent},
        "Delta_src_nat": {"mean_A": float(np.mean(delta_src_nat_A)), "mean_B": float(np.mean(delta_src_nat_B)),
                          "L_stability": L_src_nat, "U_stability": U_src_nat},
        "Delta_src_cent": {"mean_A": float(np.mean(delta_src_cent_A)), "mean_B": float(np.mean(delta_src_cent_B)),
                           "L_stability": L_src_cent, "U_stability": U_src_cent},
        "Delta_spec": {"mean_A": float(np.mean(delta_spec_A)), "mean_B": float(np.mean(delta_spec_B)),
                       "L_stability": L_spec, "U_stability": U_spec},
        "per_entity": {str(eid): {
            "A": entity_data[("A", eid)], "B": entity_data[("B", eid)]
        } for eid in range(24)},
        "call_count": len(laws),
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "manifest_hash": manifest_hash,
        "resample_index_hash": resample_hash,
        "checkpoint_hash": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest(),
        "science_seconds": time.monotonic() - science_t0,
        "total_seconds": time.time() - T0
    }
    json.dump(result, open(os.path.join(out_dir, "science_result.json"), "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    log(f"FINAL STATUS: {result['status']} ({time.monotonic()-science_t0:.0f}s science, {time.time()-T0:.0f}s total)")


if __name__ == "__main__":
    main()
