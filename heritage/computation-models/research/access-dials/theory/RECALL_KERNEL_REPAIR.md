**H2-OPEN. The single repair fails design admission: the proposed prefix directory has no bounded encoding and bootstrap for all bucket depths and counts.** The original scanner’s RESTORE-KILL remains valid; this analysis supplies neither a new failed interruption trace nor a universal sorting impossibility.

All construction and counting arguments below are **CONJECTURE, with derivations pending independent review**. The integer probe is **EMPIRICAL**.

1. **The audited repair**

   Keep the global radix traversal and retained-bit sorting kernel from [RECALL_KERNEL_CONSTRUCTION.md](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_KERNEL_CONSTRUCTION.md:115). Before changing suffixes, build an ordered navigation chain in selected words’ known original \(d\)-bit prefixes. Retain the head, tail, published-node count and current write transaction privately. Use one contiguous window of at most \(Bw\) owner words only if lossless compression actually releases additional directory space.

   The [registered ceilings](<LOCAL_PROJECTS>/new-computation-model/research/DELTA_TESTS_EXPERIMENT.md:15) remain
   \[
   S=128w^2,\qquad D=64Bw^2,\qquad
   T=64(n+n\lceil n/B\rceil)w^2.
   \]
   A helper-window version must withdraw the **at-most-\(B\)-dirty-words** headline and charge restoration of every changed helper word.

2. **Direct links explain a subcase, but leave the difficult regime open**

   Let \(a=\lceil\log_2 n\rceil\). When \(d\ge a\), a selected word’s prefix can hold its successor address. A circular chain avoids needing an extra null symbol; the private head and node count delimit traversal.

   During construction, retain the old words and addresses for the pending insertion before either owner write. Until publication, recall patches that insertion and restores the published chain. Sorting changes only suffix bits, preserving navigation fields. Logical accesses reconstruct the original prefix \(P\), and the retained-bit kernel supplies the remaining logical-prefix interpretation.

   After suffix restoration, read and retain the successor **before** overwriting each navigation prefix with \(P\). The restoration PC distinguishes that write from cursor advancement. Repeated recall continues the existing restoration phase.

   Global counts occur only when the previous bucket and helper window are restored. Radix order therefore retains the proposed sorted-output mechanism. These are candidate invariants, not an admitted elementary implementation or exact \(D/T\) bound.

   For \(d<a\), the next address does not fit. Fragmenting it across records requires locating those records before decoding the link—the bootstrap obligation itself.

3. **Compressed addresses and a helper window do not close every regime**

   Consider
   \[
   n=2^{256},\quad w=512,\quad B=c=2^{128},\quad d=64.
   \]
   Place one selected member in each of \(c\) consecutive physical blocks of \(B\) positions. Its offset within each block is arbitrary. This support family has exactly
   \[
   B^c=2^{128c}
   \]
   possibilities. Nonmember prefixes can populate the sibling of \(P\) sufficiently to make every ancestor oversized, so a bucket can first qualify as small at this depth.

   The selected prefixes supply only \(64c\) known bits. This confronts the gap between \(cd\) and the \(c\log_2(n/c)=128c\) support-information scale; absolute addresses would cost still more.

   For a nonmember helper word, exclusion of \(P\) leaves \(2^d-1\) possible prefix symbols. For a fixed window of \(m\le Bw\) such words, that constraint alone guarantees no more than the counting allowance
   \[
   F\le m\log_2\frac{2^d}{2^d-1}
     <\frac{2m}{2^d-1}
   \]
   released bits, before headers or compression state. Granting **all** private storage to the directory gives the favorable allowance
   \[
   cd+S+\frac{2Bw}{2^d-1}<65c.
   \]
   Selected words inside the window do not contribute their known prefixes twice.

   Even a simple quotient-and-scan directory fails this allowance. Granting the entire recall budget to owner reads permits
   \[
   K=D/c=64w^2=2^{24}
   \]
   reads per member. Identifying a \(K\)-position interval within the next \(B\)-position block requires a **104-bit quotient**, followed by scanning that interval. The \(c-1\) successor records therefore require \(104(c-1)>65c\) bits, before instruction overhead.

   **EMPIRICAL:** a read-only integer/rational probe confirmed these inequalities. No large owner array was allocated or executed.

   This rejects the explicit directory and quotient-and-scan encodings. **It is not a complete-state lower bound:** physical positions themselves retain support information. Other adaptive navigation or catalytic encodings are not excluded. An adaptive window choice also lacks a supplied theorem guaranteeing sufficient released space.

4. **CMR does not supply the missing admission**

   CMR Lemma 3.1 rearranges leading bits and suffixes losslessly; extraction alone releases no space. Lemma 3.2 compresses a supplied suffix code using a minimum-prefix-savings rotation and decoding tables. Direct application to \(d\)-bit symbols requires \(O(2^L)\) table words with \(L\ge d\), and neither lemma establishes this contract’s every-instruction recall ceiling. [CMR, §3](https://tmc.web.engr.illinois.edu/restore_soda.pdf)

   Here the private ceiling is only \(65{,}536\) words. Charging a smaller code, table generation, partial packing, extraction cycles and their inverses remains necessary. No such source-semantic/controller/cost expansion closes the capacity and bootstrap gap.

**Disposition:** return **H2-OPEN** to the registered broader question loop. The precise failed obligation is a promptly decodable sparse directory that fits genuinely released prefix space across all regimes and restores every construction prefix within \(D\). Implementation, smoke, code review and the registered \(n=4,6\) interruption checks remain gated.

**HEURISTIC “so what”:** known prefixes can carry candidate navigation links when they are wide enough; this repair does not establish prompt restoration for shallow sparse buckets. That is a component finding, not mission achievement.

No edits, agents, GPU work, FULL run or interruption experiment occurred.