# Source-only corpus tool guide

Use only the tools exposed in this session. The corpus is an immutable source
allow-list, not the current project folder. The project's README is setup
guidance; a corpus README returned by read_file is a different document.

If latent_memory is available, use it for topical discovery over the source
corpus before broad file scans. It returns verified source passages, not an
answer or proof that missing evidence does not exist. Choose a small k.
Use open_span to reopen an exact returned ID and follow_reference only for
literal source references. These tools do not establish global completeness.

Use list_files, search_files and read_file for filenames, exact matches,
surrounding context and complementary checks. If latent_memory is not exposed,
use these file tools alone. Cite the source identity and actual returned
coordinates; distinguish evidence, inference and unavailable evidence.
Never browse outside the configured corpus or use evaluator material.
