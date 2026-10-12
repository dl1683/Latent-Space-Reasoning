**Fix the gate sequencing before spending on this smoke.** Static review plus read-only checks of the pinned tokenizer found:

- **Names — registration mismatch; resolve before smoke.** `bindq.py:32` checks `" "+n`, not bare `n`: 148/500 selected names have **three bare tokens**. Both W1/W2 prefixes correctly add two tokens, including later facts; name endings remain token-aligned before `" box"`/`"'s"`. Thus contextual length matching works, but “exactly 2 tokens standalone” is inaccurate (`bindq.py:26–47`; registration:541–543).

- **q(n), ordering, quads — no fix needed.** Truncated prefixes end at the last name token; right-padding gathers the correct position. `query_rows`, value-state flattening, superposition grouping, and accuracy reshaping are consistently collection-major. Reordering is non-identity; renaming is injective and disjoint; bindings survive (`bindq.py:57–73,143–145,166–186,198–208,228`; `native.py:316–328`). **k=1 would loop forever**, but configured k=4/16 avoid it.

- **Histogram/statistics — no fix needed.** One baseline per base collection is valid across all quad arms because their multisets match. Differences are bootstrapped by paired collections, not entity rows (`bindq.py:225–231`; `readout.py:110–122`).

- **Leakage/selection — no blocker found.** Standardization uses training H and training-row names; epochs, reader family, decay, and layer use seen-name dev. Held-out Q is transformed without contributing statistics. Dataset namespaces separate train/dev/test; smoke uses one reader seed, consistent with its scope (`bindq.py:80–140,173–176,210–217`; config:43–58).

- **Controls — fix sequencing before smoke.** All six full closing-state sets are encoded **before** the positive control. The runner supplies no separate cheap admission stage or immediate abort after failure (`bindq.py:194–224,245–246`). The explicit-value control mainly proves colour decoding; superposition success would support reader adequacy, but averaging contextual value states does not guarantee recoverable bindings. Its diagnostic-only treatment matches registration:559–564.

- **Colour-free null — no fix needed.** Pinned-tokenizer checks show all colours and `"here"` add one token in both families. Replacement removes label-dependent token positions; labels never affect the resulting text. Semantic distribution shift limits this null’s interpretation (`bindq.py:50–53,218–224`; `q1pp.py:23–33`).

- **Adequacy/N flags — fix before smoke.** Adequacy and registered N thresholds match, but N1/N2 omit `adequate`; an `inconclusive_reader` row can still carry a positive flag (`bindq.py:232–242`; registration:562–570).

- **Memory/runtime — no demonstrated OOM; runtime protection needed.** BF16 loading and CPU feature storage make 40GB plausible; PC normalized training H alone occupies 2GiB. Each k=16 cell processes ~4.4–5.5 million block-tokens and ~92,160 optimizer steps. Long fits lack heartbeats, risking the 90-minute watchdog (`common.py:39–46`; `bindq.py:98–140,196–221`; `automon.sh:23`).

**Histogram-only h plus q-only priors cannot systematically beat histogram mode under IID colours.** Query-shuffling would strengthen attribution, but is not a smoke blocker. These remain readout claims, not causal or native-mathematics results. No files changed.