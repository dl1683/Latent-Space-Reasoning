**The sentence is conditionally licensed as a reported recipe-level stop outcome.** I found no indexing, label, or denominator bug that forces accuracy near 0.60. Raw endpoint verification remains incomplete.

- **Denominator:** `dev_n=48`; edge mode exhausts 16 selectors at k=1, giving **768 bit decisions clustered within 48 collections**. Accuracy is pooled bit accuracy, not joint collection accuracy or minimum-selector accuracy. A pooled value below 0.95 also rules out every selector reaching 0.95. [config:87](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/configs/p1_s11.json:87), [branch.py:244](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:244).

- **Labels and balance:** gold and query consistently use branch `s//16`, edge `s%16`. Exhaustive CPU enumeration found train/dev selector base rates between 0.49946 and 0.50010. On the actual frozen dev collections, even an oracle-chosen constant answer per selector scores only 0.5664. The filter therefore supplies no ~0.60 ceiling. Train/dev share pattern support; test patterns are disjoint. This limits dev generalization claims but does not explain the stall. [branch.py:64](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:64), [branch.py:76](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:76).

- **Exposure:** integer division yields 1,562 batches × 64 = **99,968 presentations**, 32 below the registered 100,000. Say “implemented endpoint,” or explicitly disclose this rounding. [branch.py:458](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:458).

- **Two timelines:** the ledger records a restart from zero and identifies the later timeline as authoritative. Last-row aggregation handles repeated keys, but cannot establish checkpoint provenance by itself. GCS reads failed because the sandbox denied access to gcloud’s credential database; no local pilot artifacts were available. [ledger:1108](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/ledger.jsonl:1108).

Replacement:

> “The ledger reports restarted seed-11 P1 pilots at r=64/r=128 finishing step 1,562 (99,968 presentations) with pooled dev k=1 bit accuracy 0.603/0.624, below 0.95; the operator recorded the stop rule as fired, remaining seeds were not launched, and no capacity claim is licensed.”

Continue the program; verifying endpoint custody is the highest-leverage next action. Ordinary bit storage is scaffolding, not native latent mathematics.

Blackboard state and synthesis calls were denied by the tool’s approval policy.