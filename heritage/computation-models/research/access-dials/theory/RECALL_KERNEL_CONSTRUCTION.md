**H2-OPEN. The specified sparse-replay construction receives RESTORE-KILL for recall latency.** The retained-bit kernel has a constructive prefix-restoration argument, and the global extension explains sorted output with at most \(B\) dirty words. Its recall scanner nevertheless exceeds the registered \(D\) bound on an explicit input.

All construction claims below are **CONJECTURE with derivations pending independent proof review**. No implementation, interruption experiment, agents, edits, GPU work, or FULL run occurred.

1. **Design and infinity-leak audit**

   I used the exact [registered contract](<LOCAL_PROJECTS>/new-computation-model/research/DELTA_TESTS_EXPERIMENT.md):

   \[
   w=\lceil\log_2(n^2)\rceil,\quad B=\lfloor\sqrt n\rfloor,\quad
   S=128w^2,\quad D=64Bw^2,
   \]
   \[
   T=64\bigl(n+n\lceil n/B\rceil\bigr)w^2.
   \]

   The construction uses finite truncated words, fixed program code, bounded explicit stacks, and a write-only output stream. It never reads the checker’s original copy. Owner words remain occupied: their unknown bits are preserved rather than treated as blank storage. The only temporary original values retained are the two words of the current swap, charged privately.

   This passes the resource *description* audit. The sparse global recall path fails the latency audit below.

   The shared incumbent is CMR §2: retain each partition bit at its position, exchange lower bits, recurse, and replay the partition to restore. Their theorem gives endpoint-restoring sorting in \(O(n\log U)\) time; it does not establish this every-instruction, bounded-recall contract. [CMR, §2](https://tmc.web.engr.illinois.edu/restore_soda.pdf).

2. **Finite machine state and elementary swap**

   Use unsigned arithmetic modulo \(2^w\). A shift by at least \(w\) returns zero; array accesses require an address below \(n\). Store the active bit as `qcode=q+1`, with zero denoting a leaf. Thus no negative index or unbounded integer is needed.

   Reserve two stacks of \(w+1\) frames: one for global radix traversal, one for retained-bit sorting. Each frame has sixteen \(w\)-bit fields. An inner frame stores:

   ```text
   lo, hi, qcode, logical_prefix, cut, multiplicity,
   zero_count, forward_swaps, restored_swaps,
   left, right, phase, next_child, emission_cursor,
   pending_child, scan_mode
   ```

   Reserve another 64 word slots for configuration, transaction values, scan registers, depths, address arithmetic, and cursors. Reserve 128 control bits for PC, saved PC, transaction/control phases, and the recall latch. Private addresses wider than one word use counted control fields and expanded address instructions.

   The resulting allocation is exactly

   \[
   32(w+1)w+64w+128
   =32w^2+96w+128\le128w^2
   \]

   for every registered \(w\ge4\). There is no host recursion stack or dynamic undo list.

   At partition bit \(q\), let \(L=2^q-1\). A swap expands into:

   ```text
   x     = READ A[left]
   y     = READ A[right-1]
   delta = XOR x,y
   delta = AND delta,L
   nx    = XOR x,delta
   ny    = XOR y,delta
   WRITE A[left],nx
   WRITE A[right-1],ny
   ADD forward_swaps,forward_swaps,1     # commit
   ADD left,left,1
   SUB right,right,1
   BRANCH scan
   ```

   Address formation is separately expanded. Each displayed read, write, Boolean operation, arithmetic operation, and branch is an elementary instruction. The transaction preserves bit \(q\) and every higher bit at both positions.

3. **Kernel and arbitrary-prefix restoration**

   The partition scans inward using the retained bit: skip left zeros and right ones, then exchange lower suffixes. Its completed swap pairs are disjoint. Because their selection bits never change, replaying the scanner reconstructs exactly the same pairs.

   The explicit recall cases are:

   | Interrupted phase | Restoration action |
   |---|---|
   | Before either swap write | Discard the unfinished private calculation. |
   | After the first write | Write saved `x` back to the left address. |
   | After the second write, before commit | Write saved `x,y` back to both addresses. |
   | After commit, including pointer updates | Include that pair in the committed replay count. |
   | During a child computation | Restore the deepest published frame first, then its ancestors. |
   | During inverse replay | Continue its existing PC, transaction, and restored-pair cursor. |
   | During emission | Abort output; restore outstanding frames. |

   A child frame is initialized completely before publishing its stack depth; it cannot mutate the owner array before publication. A frame is popped only after its region is restored. Interruptions during those metadata instructions therefore retain every outstanding restoration obligation.

   The first recall latches abort mode and preserves the interrupted PC. Subsequent recall requests leave the restoration PC and progress unchanged. They never restart replay. An unlimited stream of external requests is not counted as algorithm instructions; the initial dispatch is charged.

   Kernel pseudocode:

   ```text
   SORT_REGION(lo,hi,q,p):
       if region is empty: return
       if qcode == 0:
           EMIT p once per member
           return
       partition using retained bit q; save cut and completed-pair count
       SORT_REGION(lo,cut,q-1,p)
       SORT_REGION(cut,hi,q-1,p OR (1<<q))
       replay completed pairs; return

   RECALL:
       latch abort; suppress future EMIT
       patch any uncommitted forward swap
       continue an inverse already in progress
       otherwise undo published frames from deepest to shallowest
       finish when no owner mutation remains
   ```

   For a **contiguous region of at most \(B\) words**, the replay scans at most \(B\) positions per active level. A proposed elementary expansion gives

   \[
   R_{\mathrm{local}}\le32Bw+96w+32\le64Bw^2=D.
   \]

   This is a construction-level instruction budget pending expansion/review, **not an admitted implementation bound**.

4. **One coherent global extension**

   Traverse the value-prefix trie in increasing radix order. Count each visited prefix by scanning the restored array.

   ```text
   VISIT(prefix,depth):
       c = count matching owner words
       if c == 0: return
       if depth == w:
           EMIT the complete value c times
       elif c > B:
           VISIT(prefix·0,depth+1)
           VISIT(prefix·1,depth+1)
       else:
           retained-bit sort the matching positions
           EMIT their logical values in order
           restore those positions before returning
   ```

   Inside a small bucket, partition scanners skip positions outside that bucket. Bucket-prefix bits remain unchanged, so membership remains readable throughout sorting and restoration. Each recursive child uses a physical subinterval of the bucket’s selected-position sequence.

   This supplies the missing **global output mechanism**: earlier buckets are fully emitted and restored before later buckets begin. Large duplicate buckets reach a complete-value leaf and emit their multiplicity without mutation. At most one bucket is transformed, so the current dirty footprint is at most \(B\).

   Let \(h=\lfloor n/(B+1)\rfloor\). There are at most \(h\) oversized prefixes at each depth. The small-bucket leaves satisfy

   \[
   \sum_{\text{leaves}}(w-\text{depth})\le h\,w(w-1).
   \]

   A proposed compilation envelope is

   \[
   W\le
   16n(1+2wh)+48nhw(w-1)+96n(w+1)+128(w+1)^2.
   \]

   Its algebraic upper bound fits registered \(T\). **The instruction-envelope constants remain CONJECTURE until a fixed elementary source is expanded and reviewed.** The recall defect below already rejects this version.

5. **Exact global recall-latency witness**

   Choose

   \[
   n=2^{40},\quad w=80,\quad B=2^{20}.
   \]

   Let \(H=2^{78}\), and set:

   ```text
   A[0]       = H+1
   A[n/2-1]   = H+2
   A[n/2]     = 3
   A[n-1]     = 4
   every other A[i] = 2^79
   ```

   The high-bit-zero bucket contains exactly four records, hence qualifies as a small bucket. Its first partition, at bit \(78\), has retained bits `1,1,0,0` and commits pairs

   ```text
   (0,n-1), (n/2-1,n/2).
   ```

   Request recall immediately after the second pair’s commit. The specified inverse replays from the region boundaries. After undoing the first pair, its left scanner must read every nonmember position from `1` through `n/2-2` before reaching the second pair.

   That scanner alone performs at least

   \[
   n/2-2=549{,}755{,}813{,}886
   \]

   owner reads, while

   \[
   D=64Bw^2=429{,}496{,}729{,}600.
   \]

   It exceeds \(D\) by **120,259,084,286 reads**, before charging other restoration instructions.

   **EMPIRICAL arithmetic smoke:** a read-only integer calculation confirmed these numbers. No owner-array execution or interruption enumeration was performed.

   This is a latency witness for the specified replay routine. It is **not** a complete-state collision or an impossibility theorem; the saved endpoints could support shortcuts on this particular input.

6. **Repair disposition and actionable gap**

   No repair was attempted. Caching every selected address would close the scanner defect, but a straightforward \(B\)-entry address table requires

   \[
   B\lceil\log_2 n\rceil=41{,}943{,}040\text{ bits}
   \]

   at this \(n\), exceeding \(S=819{,}200\). Caching only the four witness addresses would not close the general defect.

   The precise remaining obligation is a **sparse support-navigation/inversion mechanism** that restores every elementary prefix of a bucket with up to \(B\) arbitrarily located members within \(64Bw^2\) steps, while fitting the complete \(128w^2\)-bit private allocation. Any metadata placed in owner words must preserve their original information and itself be prefix-restorable.

   Keep the registered interruption experiment gated until that obligation has a coherent construction. The local kernel remains a candidate component; it does not earn H2-ALIVE or same-contract CMR absorption.

   **HEURISTIC “so what”:** bounded mutation can support global sorted progress in this construction, but promptly finding the sparse locations to undo remains unresolved.