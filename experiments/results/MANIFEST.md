# experiments/results — deletion manifest (2026-09-05)

Raw dumps of closed lines were deleted from `experiments/results/` so that the
tree holds only verdict-bearing summaries. Every deleted file is listed below
with its byte size and SHA-256 so a regenerated artifact can be checked against
the original. Tracked files are recoverable from git history at the commit
recorded for the line in `experiments/ledger.jsonl`; files marked `untracked`
were git-ignored (weights, caches, row-level evidence) and are regenerable only
by re-running the runner recorded in the ledger.

Rules applied:

- Everywhere: `*.npz *.pt *.pth *.ckpt *.safetensors *.pkl *.npy *.gz`,
  `checkpoints/` directories, and any file larger than 2 MB.
- Closed-line directories (every directory except `write_boundary_v1`,
  `commit_seal_rung1`, `ceg_1`, `crc_0`, `crc_1`, `svb_0`, `svb_1`, `svb_2`,
  `svb_qwen3_*`, `curvature_*`): per-run / per-seed / per-row dumps
  (`lm_dyn_v1/analysis_<tag>.json`, `evidence.json`, `seed_*_rung_*.json`,
  `predictions.jsonl`, `cases.json`, `eval_rows.json`), smoke-run outputs,
  logs and checkpoints. Kept: `result*.json`, `results.json`, `*_result.json`,
  `verdict*.json`, `summary.json`, `analysis.json`, `manifest*.json`,
  `config.json`, small preflight / lock / eligibility records, `*.md`.
- `commit_seal_rung1/partial.json`: stale output of an older runner version.
- Directories left empty by the above were removed: `delta_cap_test`,
  `fba_0_cap_test`, `fba_0_delta`, `necessity_navigator_v1`,
  `pipeline_smoke_8`, `ppo_bptt`, `svb_qwen3_eligibility`,
  `svb_qwen3_idempotent`. `commit_seal_rung1/` was recreated empty for the
  running rung-1 runner.

Size of `experiments/results`: 225,281,775 bytes before -> 21,725,606 bytes
after (134 files, 203,556,169 bytes deleted; 103 tracked, 31 git-ignored).

| path | bytes | sha256 | git |
|---|---:|---|---|
| `band2_decisive/band2_dists.npz` | 31378 | `bc3766278bece5878558b655c6cebd905e0f9e3576cdb7b9a28a3e21a7b65ff7` | tracked |
| `commit_seal_rung1/partial.json` | 11685 | `559f543e7ec7640088c4b336613b36f1280b46e2461c7ce2af768fdc53bcd919` | untracked |
| `commitment_hysteresis_v1/evidence.json` | 52554 | `80148ba0dd9704ee0f7a4583f176379fb813494841e8017e90003f60ca56e781` | tracked |
| `delta_cap_test/fba_s0/ckpt_500.pt` | 366329 | `213aeca7bfe7a2122cd86d4ac209f1a9ad759565839116fcb4cbbce95a8f4c38` | untracked |
| `delta_cap_test/fba_s0/final_model.pt` | 119505 | `61906b547f37bcd157b9d13bcff04f75b14d04b73a4890a12249dfb707d4f8d0` | untracked |
| `delta_cap_test/fba_s0/log.json` | 4209 | `728754e4dec878fdeb2b237a2ea8f30c740f2c655d5600b953c6fc89daba97cb` | tracked |
| `endogenous_action_carrier_v1/evidence.json` | 1859 | `249daec79f28ce53369487b09c4a8829725bf4aabe8c7f952a2e847c574f3da1` | tracked |
| `endogenous_action_carrier_v1/model_s137.pt` | 1440056 | `7a77164cce64efe22570e5ecafb0abbc5020bf839a2d519b11d0965f471272ef` | untracked |
| `endogenous_action_carrier_v1/model_s2026.pt` | 1440069 | `549acb4f94e3bcc924c99ef0237ae4d59867aa80585da766ef2b559ac1f4d18c` | untracked |
| `endogenous_action_carrier_v1/model_s42.pt` | 1440043 | `5f58eb78fcec1f69ac4e3b6ddfdd0d5b372578368e36a68314d7d51cd8b3b7c6` | untracked |
| `endogenous_response_quotient_v1/evidence.json` | 12192 | `3563da7e948d9fdb96ab07bea25ebc24634abaf20631f6f0ac94d9c7f83bc56f` | tracked |
| `execution_mode_invariance/emi_checkpoint.json` | 85 | `dd3967bd2f4c76b8f4d79a31bd2887983fec8dc19ecb90424d5c68383e20cf3a` | tracked |
| `execution_mode_invariance/emi_core_dists.npz` | 94302 | `7386f1330ea3fa4498bd86ad6010f8d5405de5605bc043574f579c6a872fedc7` | tracked |
| `fba_0_cap_test/fba_s0/ckpt_1000.pt` | 353759 | `ba1ae995aa414aa5f5c321c4b881dc879db33bd30f835cd550acaf81ac9b8f77` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_2000.pt` | 355679 | `34a0c9a4bc51c97b59648bb32aa8bf713a038b021c9fd17f50892c32e1777d82` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_3000.pt` | 357663 | `1c4c9ba9b1324d8684bf77a242ff2df4d1fbbe88e0081bbd63079b91a78b51f1` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_4000.pt` | 359647 | `2d9b0c90bc52271d99151fcde22e513bf1684e02582d02a64ba4ed56c3999b2d` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_5000.pt` | 361631 | `ab52c3630874eb1cd8a9689548e7256b56a790c48e474db26bd806898fbf28a8` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_6000.pt` | 363615 | `d62e3670e3d1a16381166aaa71c73996b83bcad0d5f26c151f89d9f49cfed7f0` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_7000.pt` | 365599 | `03cd554af041507134eeff38b680ab7237e26639d1a42a29b7e6fc1a8baf66de` | untracked |
| `fba_0_cap_test/fba_s0/ckpt_8000.pt` | 367583 | `3be9f1a5b946358a42221b8438c2af0ca0bbb21c3dff2149698b1ce4124bf0b9` | untracked |
| `handle_mu/seed_137_rung_1.json` | 94403 | `46b2ebce3f0dc96c58f5709736dc26ce06d6aa37412f3b52e0f402c17d53939b` | tracked |
| `handle_mu/seed_2026_rung_1.json` | 85650 | `1a628716844bcdb804610a72e70402f6e44879c210556923c2ff23bb28a42a58` | tracked |
| `handle_mu/seed_42_rung_1.json` | 84088 | `d4dba135da389e8c8fcb8040952238721daac8aaffec9bdfdb727d45714c894e` | tracked |
| `handle_mu/smoke_v2_seed_42_rung_1.json` | 67254 | `6d98fca22a4128eaa7f8d7fc7f8954b403ef353e8c66a40d5eb906218107be2a` | tracked |
| `handle_mu/smoke_v2_verdict_rung_1.json` | 79028 | `a9b78d69009f128566175b219c4fb4aefdfbe4afccfbe0ad30cc153596238460` | tracked |
| `handle_mu/v2_seed_42_rung_1.json` | 316304 | `d611b84704a7811f06c28677c18f58df76914f43a35877b8c779d8f756675d6e` | tracked |
| `lac0_run_20260901_063614/typed_seed42.pt` | 2972369 | `ffec6011df0b48fe00de020e8dd528337a0ff450d0004e4c58186a9e6caebdf5` | untracked |
| `lac0_run_20260901_064645/typed_seed137.pt` | 2972415 | `20f7bbf3ba91823b9d113b56c17dd92b561f7c05df06fd8701df028a635f118a` | untracked |
| `lac0_run_20260901_064645/typed_seed2025.pt` | 2972461 | `f54bdafd9972d93dd349fc382442584fe27596b1991b9ea5a71d15857c2531ff` | untracked |
| `lac0_run_20260901_064645/typed_seed42.pt` | 2972369 | `094ac3444ddc0a91a3c21ab83621fa300447c4915526a901fc1be7c6499a916a` | untracked |
| `lac0_run_20260901_064645/untyped_seed137.pt` | 2941212 | `38adc70c554b41c8fef51de7093883bcf6f548e8a812fae79429b054e468d19c` | untracked |
| `lac0_run_20260901_064645/untyped_seed2025.pt` | 2941271 | `cdc3928a8b048f332dd423fb17dcc799d123bd6fd42ae00ed71152997187a827` | untracked |
| `lac0_run_20260901_064645/untyped_seed42.pt` | 2941153 | `b3f44cd22281a2f21bc06309f0d186bd8a9281093ce03bc22ab93aa86515397d` | untracked |
| `learned_action_carrier_v0/typed_seed137.pt` | 2972415 | `d4668e549d82e4dd560c7f8415d053d1627dcf041f064582037b48a4e193640c` | untracked |
| `learned_action_carrier_v0/typed_seed2025.pt` | 838761 | `84b64d89ad768a433aa34e9dcd2ad499d7292d9c47308674866059cd128bd612` | untracked |
| `learned_action_carrier_v0/typed_seed42.pt` | 2972369 | `83c84150fb878e7e21577b71bf4e4238f22ae09b8b9a550d13c0d1793dd4fc29` | untracked |
| `learned_action_carrier_v0/untyped_seed137.pt` | 2941158 | `0951f298b64286cfdce980f3e2eb7747c92e713afe6ebaf2dbbf109e423092f9` | untracked |
| `learned_action_carrier_v0/untyped_seed2025.pt` | 2941219 | `7373897702885ddc6db9b4ebc3caec1ce2330644f14924c7b94e0963b4d9a07a` | untracked |
| `learned_action_carrier_v0/untyped_seed42.pt` | 2941097 | `a4c22f2f7497c16bfdbaf6021d1c95eb0e573fc8f140107f09472f9c9bbeea74` | untracked |
| `lm_dyn_v1/analysis_base.json` | 286613 | `3c8905f5e311b05450c0e60695dddc1fcd82b65afb039d19a3c93070a9382930` | tracked |
| `lm_dyn_v1/analysis_basesmoke.json` | 49795 | `cc631378232c4900b690a857c885da2ef0b947c2e76e5d1ed72a69321a06c6b6` | tracked |
| `lm_dyn_v1/analysis_ctxoverlap_A.json` | 5153756 | `0f386eaed68ffdee305cae33da13ccb53fdd7af6749891de2839110280f68efb` | tracked |
| `lm_dyn_v1/analysis_ctxoverlap_B.json` | 5153493 | `d12ecb8c83c11388dfda8176909f88fbaac4af39c76f4e7cee56829f5a325ce2` | tracked |
| `lm_dyn_v1/analysis_delta.json` | 191672 | `861a08cd0a1d3a8ec6f3872b5ead377474cde1d1095f4393607ad390c2241359` | tracked |
| `lm_dyn_v1/analysis_deltasmoke.json` | 39349 | `3b6dd3611f754475eccccedb4ebf143ab580eadcaa806238af3b06a4da078f2b` | tracked |
| `lm_dyn_v1/analysis_ext.json` | 76869 | `b273e8e465afc57573c374c38e962b5cc7afe77484ab71f2776227223db19010` | tracked |
| `lm_dyn_v1/analysis_fwdA.json` | 242361 | `1140a7d78da00105cb1683f190ee5c8e89b0e1c7fd7c050603ea841ed26f9d1b` | tracked |
| `lm_dyn_v1/analysis_fwdB.json` | 242338 | `ba7b7ce335f6301dc4d1242981b2bf1f279d5b4d0914a520c8808608ac191ca0` | tracked |
| `lm_dyn_v1/analysis_fwdsmoke.json` | 48119 | `4c359d733d363fb635302809531e44983d3b1126ef41573a8827c042adb6c7af` | tracked |
| `lm_dyn_v1/analysis_locoA.json` | 387936 | `ca8c1d1cad2b5be50b72399c6fea8f9251c7565499e48d464aef345db08fd12d` | tracked |
| `lm_dyn_v1/analysis_locoB.json` | 388094 | `4059f1f0b81551cab9cd61e2b6ced11e8a4959dbc9d68ca16fc964b2b50261ab` | tracked |
| `lm_dyn_v1/analysis_locoeq2A.json` | 492778 | `9471977d1092c4c47793f2ce57b0b4a38db2ee74956d0ad3e3c0248e7ca298e9` | tracked |
| `lm_dyn_v1/analysis_locoeq2B.json` | 493356 | `f526d712b5b35444ab511d43dbc4be36666dd27e4c7ff2566cae1dc681559709` | tracked |
| `lm_dyn_v1/analysis_locoeqA.json` | 491473 | `f5255ceaf73975503c4b3e4d63b1c8839079363325a0958de0153b110f17eae8` | tracked |
| `lm_dyn_v1/analysis_locoeqB.json` | 491438 | `68e1036932ab9e891393bf0ed2ee4e9968ef365af480cc89a7acaceb87401c26` | tracked |
| `lm_dyn_v1/analysis_parity_ref_A.json` | 9681960 | `a0c220eddb1693fe0666d11f23967dd35d35214b9e5e29065fb50cafcf594071` | tracked |
| `lm_dyn_v1/analysis_parity_ref_B.json` | 9656180 | `6012f33f5221726a8a294b94225d440e8ddb020f3f3f0f51fb79c9cf04dd5d1c` | tracked |
| `lm_dyn_v1/analysis_resAA.json` | 1215185 | `f989d653c3263618e5e743a76b2cf71d26b026227371b28e0370dd4f2ed97eed` | tracked |
| `lm_dyn_v1/analysis_resAB.json` | 1215168 | `dc75491b051f2c1c4a3c3040b2741f6c2053c514e25c84b54bbd4d5d14383ea9` | tracked |
| `lm_dyn_v1/analysis_resSA.json` | 1151163 | `5cb24535fe53f5b7bb2cee01e7e83a6f12ca2cf242732bcb78a4c7d5721aa143` | tracked |
| `lm_dyn_v1/analysis_resSA2.json` | 1215404 | `50a00af3a630c6895bd8a16cdf11bb97b8c55dbdccc47888aa22b3020595f9db` | tracked |
| `lm_dyn_v1/analysis_resSB.json` | 1215676 | `d2582747e59c5bbe909fb1cc00f7777f3d76b9edd44444a097683b5afb6958f7` | tracked |
| `lm_dyn_v1/analysis_residsmoke.json` | 231461 | `a4e3f2ae72d852df4c8c2c55e9f1a7176229ca18d20a6a9f8f84865f61e77ba7` | tracked |
| `lm_dyn_v1/analysis_slot.json` | 213514 | `3d5540fcc20eea04ae1c0c4a07b6c7669ea25c008e612efedb643281eed0ced0` | tracked |
| `lm_dyn_v1/analysis_styleA.json` | 321521 | `2cb4b59b97a9bd42cff4cccc3271c237e3fd533435399e8e40131b176977c62c` | tracked |
| `lm_dyn_v1/analysis_styleB.json` | 321647 | `94aa88cd01a23e11b9064cbd42c57c90e47398429775f7d80db944f002b1e450` | tracked |
| `lm_dyn_v1/analysis_stylesmoke.json` | 65370 | `d18b5e69d15d969302f434e01140726090927714fd12f2a96c2d5df4e07de756` | tracked |
| `lm_dyn_v1/analysis_unseenA.json` | 658488 | `6a78a5617c74c9c0b69c7c7b8676edf8a6f231efe6248fe0e4520222f5ab1408` | tracked |
| `lm_dyn_v1/analysis_unseenB.json` | 658840 | `9cf5c5db26a643c97b52dc88882c521d11fbc51d6a98ac5ffa735afa7d62e706` | tracked |
| `lm_dyn_v1/analysis_unseensmoke.json` | 128132 | `1e767bc9465f62a73c4a38dc02fbc02b3e274eff43995ddd24533a3a29e6ba01` | tracked |
| `lm_dyn_v1/round34a_evidence_ctxcapA_raw.npz` | 239149 | `7d2406e3545fa9f3797f6d948553e02606895176161c54b8e2ecc4bcc961f1cb` | tracked |
| `lm_dyn_v1/round34a_evidence_ctxcapA_static.npz` | 239340 | `5f7bf3287ea6741a5745642c2553f2330851e6dc9158ff8b6e380d4e10e5c60e` | tracked |
| `lm_dyn_v1/round34a_evidence_ctxcapB_raw.npz` | 239023 | `4cfe51d601b4cb1027e81b50985b9b1df96b59f2a505d442579d7082cccc8df6` | tracked |
| `lm_dyn_v1/round34a_evidence_ctxcapB_static.npz` | 239659 | `c808ffd7602abb9cb2e59c32327811ea1f2512db209a49b58a036eddd30738b9` | tracked |
| `lm_dyn_v1/round34b_evidence_ctxoverlap_A.npz` | 854192 | `8a45da4ce402ca8bc8ad9cef1059591d4a065a0545292177b8d55bd22e3a0659` | tracked |
| `lm_dyn_v1/round34b_evidence_ctxoverlap_B.npz` | 854758 | `7ff08cc8a09b280dd0896c6a6ac12c7ecab5e62c07db45f1962848782091faa9` | tracked |
| `lrb_decisive/lrb_dists.npz` | 20918 | `b24d83da1701f42812140b379e701fdedb85412f5d04c03674988ee3067d6524` | tracked |
| `native_bridge_v1/resample_index.npy` | 384128 | `f92f29034d0aedbd96841d7ba3fed7acda401571d36623ffe23c00388a69fd54` | tracked |
| `native_bridge_v1/smoke_result.json` | 3742 | `6cf9b10fa895ef0999d749ce04c56619918528292e86cf29ea01bf7805b507b2` | tracked |
| `necessary_register_rung1/consumer_seed11.pt` | 1149550 | `8ecc57fe31db3214fef5592ad6de598b226fd8b2fbd2c9407edd2d47687b113f` | tracked |
| `necessary_register_rung1/consumer_seed23.pt` | 1149550 | `6cb057cffeb61d7637de0cb16e510209d4e755495b5f8492eca940da7569af6b` | tracked |
| `necessary_register_rung1/consumer_seed37.pt` | 1149550 | `e3e6cc18a74f5fbfe019a0e9a420217ecc684f011d005950ab24728a547e8984` | tracked |
| `necessary_register_rung1/run_oracle_crossed_rows_seed11.json.gz` | 111307 | `04f9069e55d01a3ee5b780155c79a933044738737f07bcd9418ec5f294d677b9` | tracked |
| `necessary_register_rung1/run_oracle_crossed_rows_seed23.json.gz` | 102208 | `2113c931f23dc533f24b40421c39f829e8e79d1ff56bd3e082bd5300ce97fcf5` | tracked |
| `necessary_register_rung1/run_oracle_crossed_rows_seed37.json.gz` | 107839 | `4483b0996394612b7354dfb5784a6d0c691a491193b5596ab9c6ff113f407b88` | tracked |
| `necessary_register_rung1/run_rows_seed11.json.gz` | 153365 | `0d4e51278a2e4c11cad3790d46d83c294d02fee57ee1340a5832ed82cb89f852` | tracked |
| `necessary_register_rung1/run_rows_seed23.json.gz` | 144151 | `380e47bcf7b4df43b7a1a4284f5962ebc63c991f2a3993906475d4ca5453bffb` | tracked |
| `necessary_register_rung1/run_rows_seed37.json.gz` | 148177 | `9ec6e969c2bd721ead1bdc3ca8aad9a8efbb85e97a615c73e60bc77d7d693e99` | tracked |
| `necessary_register_rung1/writer_seed11.pt` | 375837 | `14eab04ab6f0edc9769f5436adfd3042ab78162099ed38e3a7a9e8cff7e5989a` | tracked |
| `necessary_register_rung1/writer_seed23.pt` | 375837 | `4bdba6f1abfbcda56066142e6d4d481f1c836e5f7e202c5d1fdcf65c60635921` | tracked |
| `necessary_register_rung1/writer_seed37.pt` | 375837 | `7fc3c214d6b48e3690c436b7b02a21b401ed85c11a16e5ed709cd4da5af2f261` | tracked |
| `necessity_navigator_v1/smoke_result.json` | 3502 | `6a4fa3ed707491085bc0356ca39850ede90756de5ea00cc3e6e0f5be44c3707c` | tracked |
| `nlm001_v1/HuggingFaceTB__SmolLM2-360M.npz` | 16688658 | `a922e96d78d0c329df6306dff9600bb2ecca23c45692370c906cd4b40751b92d` | tracked |
| `nlm001_v1/Qwen__Qwen3-0.6B.npz` | 17742946 | `1fdf97dc2166770ccd9f56ada62740d4be203593933b48524953c33bd0dac2d7` | tracked |
| `nlm001_v1/google__gemma-3-270m.npz` | 11325824 | `b86255c1cd564713569fff83e7bde3b225e93afbc9589aeb73a4e92455e81b5e` | tracked |
| `observational_selectivity_quotient_v1/evidence.json` | 79952 | `842bccba09d403f58837a64c8e647c54304c24f6c6617d4778c24ebdf2fa0077` | tracked |
| `onewrite_recall_rung1/iface_seed11.pt` | 264462 | `13246e53d6a004f1d43d575e7556340dfa909b1c30a74c4bfb4ebd20885ae659` | tracked |
| `onewrite_recall_rung1/iface_seed23.pt` | 264462 | `48696c000624acbe1c764e0fee0ccfe56a92fd41af25ff247b4460a8963ca495` | tracked |
| `onewrite_recall_rung1/iface_seed37.pt` | 264462 | `14bb50b220589feae561ead1dace44fef8d969fa0a8b1543ca22d187da280260` | tracked |
| `onewrite_recall_v1/iface_seed11.pt` | 264462 | `241af3361d694ff244ba44937aeb12f42c225e6a65429ea9a208aa0071ea4eb2` | tracked |
| `onewrite_recall_v1/iface_seed23.pt` | 264462 | `a47c47dbcb4ac936d0fb95dd5061229a5cde62ab00a569541e54e937008d1a20` | tracked |
| `onewrite_recall_v1/iface_seed37.pt` | 264462 | `38ed42732f02098e3b0f3969946aafaacc5ae240d3dbdfb27b5c5661ef72e827` | tracked |
| `onewrite_state_v1/smoke_result.json` | 55144 | `7af40f3f34a7b280bb6a830c3552b25072db8fa1fdeabd0008284e7461fa45be` | tracked |
| `operational_quotient_v1/weights.npz` | 108588 | `c6cb96496aebd0a91e840cf22d0122e62129edf2fe555869b07dc568ab5535b4` | tracked |
| `oracle_actuator_rung0/J_seed11.pt` | 132720 | `4a3d15423b53729639db78aa4f912342803fb9ef883cefcc8af3ed087989972d` | tracked |
| `oracle_actuator_rung0/J_seed23.pt` | 132720 | `c4efa4f09509ec0780b9e9d0cf964635393af5e11bb984963174cf3a93316b23` | tracked |
| `oracle_actuator_rung0/J_seed37.pt` | 132720 | `6bd0c05f5942309ae218c59b3d944a0f60f35fbc441408ca1b2179a19645aeab` | tracked |
| `path_fiber_v1/obs_cache.npz` | 94630 | `5709ebb18f72d299eafcc44b00cedbc084b3cc091a84fb94cd11491c283d4b41` | untracked |
| `path_fiber_v1/predictions.jsonl` | 205605 | `77d04f7a30cca7405e6c99ec07d643f2863938b62afc88df07ce3281d344f59b` | tracked |
| `permutation_eligibility_v1/cases.json` | 47378 | `55e3c577ae5c76b3e55a00214c2726e6601a5a9ce40c2483d5ff868b4a67d84d` | tracked |
| `pipeline_smoke_8/Qwen__Qwen3-0.6B.npz` | 1760351 | `e015704dbd557a9193320cbf8de6ef34e3f3005b776cf9b8187c9ba9c2f8fbb0` | tracked |
| `pipeline_smoke_8/analysis.json` | 6019 | `f11936384d332c7ce59bb1c9ed76d40c12665fd05c7782b9768cacabe7bd441b` | tracked |
| `pipeline_smoke_8/manifest.json` | 755 | `9e0fa6340133255baf79349f860c6f9013af695d813cc1743ccc97659fc5fda9` | tracked |
| `psq2_v3_d4/d4_matrix.npz` | 34308 | `5f476606e5df78579308e2aed163e8d57fdba4a5a3d57eb3f2d447e6d54440e9` | tracked |
| `psq2_v3_d4/gt_d4_matrix.npz` | 33042 | `780388178588eb9005376747c18ffaaf5660104c7c7414edb1fbc86bb79eef4d` | untracked |
| `psq2_v3_d4/hidden_states.npz` | 1575178 | `bb35321e3d4416b87a7368660c9dfc8bf796e71f1cb47e6fbec286fed2b67f36` | tracked |
| `psq2_v3_d4/response_table.npz` | 350220 | `141fc6cd795b172407f5b737e4bf62702b91475ba801a2d4d6bbadc8cef24f06` | untracked |
| `psr_v2/eval_rows.json` | 47780 | `3590fb07191b940791c70c4aaa360ef719e81498e4f2c63287854402e7293e73` | tracked |
| `psr_v2/sig_cache.npz` | 45769 | `4451276ca4325388a8c4b87a134a14d4d770254d691da1f00713a5ca6bd93a0f` | untracked |
| `query_port_composition_v1/evidence.json` | 40181 | `366c88b3ae4618deabd56f1f6674b7d72b53332b25cb1d0c32d62b100e3c1c16` | tracked |
| `register_bridge_preflight_v1/run_rows.json` | 2964178 | `519846614554b6f045038fee91498638f83957ff7fd0163781131f5f9e787095` | tracked |
| `right_continuation/right_cont_dists.npz` | 10486 | `a56926c265868342848162b696026c7437827be987c987f0da7dab2fe8a9a3c1` | tracked |
| `signature_restatement_v1/phase4d_results.json` | 5411771 | `d8866ba8408748a5dd806160ec7bb7a3cc7d6714a02e6eaff5d9711013e362e4` | tracked |
| `state_bus_v1r1/bus_seed11.pt` | 396665 | `e1cd24e6d9d2c805a27416a28d379fb4627d4e9cdf37be2311be2179544ada39` | tracked |
| `state_bus_v1r1/bus_seed23.pt` | 396665 | `4faecf530620fe3a7a1749cfc0e493e784acc8517db73085a7021c2313eb9f47` | tracked |
| `state_bus_v1r1/bus_seed37.pt` | 396665 | `4cb1d608af90d74adc2fb9674e11e4e9f5cc499dc0fed5fb1a70e703b49b3fab` | tracked |
| `svb_1/obs_checkpoint.npz` | 146038 | `affa07ec9fd55cd0c5bf036a20d810a541182da78d33af894428e2c87d8e97e2` | tracked |
| `svb_2/obs_checkpoint.npz` | 255550 | `28bc2846c823420394b6922c26b33d4a93bec224503c8e6f473f0a8886918e89` | tracked |
| `svb_qwen3_depth5/obs_checkpoint.npz` | 91282 | `2830998713d49e8a46642954ae1ac49d1e815bc35479367c262c1543e14d6d7a` | tracked |
| `svb_qwen3_formal/obs_checkpoint.npz` | 146038 | `46207a2d84bee5f2d584a83d1dc7baaffcc97f916a8174513afe94380fc04ef6` | tracked |
| `vision_cifar100_dinov2s/cache.npz` | 11574622 | `8de4f0b0c47e1272de8c948c770517a66edd7beff933ccc29d995886c8145791` | tracked |
| `vision_cifar100_dinov2s/pixels.npz` | 22436403 | `0da05f31cd6ccec4bd4de5f33b4577823278d42c7198f8f983f8630d481ac07d` | tracked |
| `vision_cifar100_randinit/cache.npz` | 11555340 | `d5bd67d3de5cea77b7a132a2175aab6bd40b4afff3e677ee7306f471b2c079aa` | tracked |

## Deleted configs (2026-09-05)

Removed from `experiments/config/` as configs of closed lines with no remaining runner; each is recoverable at the `git_commit` recorded in its `experiments/ledger.jsonl` entry (last tracked at commit b70c861).

- `experiments/config/commitment_hysteresis_v1.json`
- `experiments/config/control_cost_v1.json`
- `experiments/config/coordinate_v1.json`
- `experiments/config/coordinate_v2.json`
- `experiments/config/coordinate_v3.json`
- `experiments/config/endogenous_action_carrier_v1.json`
- `experiments/config/endogenous_response_quotient_v1.json`
- `experiments/config/fusion_fission_v1.json`
- `experiments/config/interchange_v1.json`
- `experiments/config/interchange_v2.json`
- `experiments/config/learned_action_carrier_v0.json`
- `experiments/config/lexical_probe_fresh_v1.json`
- `experiments/config/lexical_probe_fresh_v2.json`
- `experiments/config/lexical_probe_fresh_v3.json`
- `experiments/config/lexical_probe_fresh_v4.json`
- `experiments/config/lexical_probe_v1.json`
- `experiments/config/native_bridge_v1.json`
- `experiments/config/necessary_register_rung1.json`
- `experiments/config/necessary_register_v1.json`
- `experiments/config/necessity_navigator_v1.json`
- `experiments/config/observational_selectivity_quotient_v1.json`
- `experiments/config/onewrite_recall_rung1.json`
- `experiments/config/onewrite_recall_v1.json`
- `experiments/config/onewrite_state_v1.json`
- `experiments/config/oracle_actuator_rung0.json`
- `experiments/config/path_fiber_v1.json`
- `experiments/config/permutation_eligibility_v1.json`
- `experiments/config/pmo_0r.json`
- `experiments/config/psq1_d4_8b.json`
- `experiments/config/psq1_v1.json`
- `experiments/config/psq1_v1_8b.json`
- `experiments/config/psq1_v1_8b_instruct.json`
- `experiments/config/psq2_v1.json`
- `experiments/config/psq2_v2.json`
- `experiments/config/psq2_v3.json`
- `experiments/config/psq2_v3_d4.json`
- `experiments/config/psq3.json`
- `experiments/config/psq3_alpha.json`
- `experiments/config/psq3_micro_cpu.json`
- `experiments/config/query_port_composition_v1.json`
- `experiments/config/rcq0_v1.json`
- `experiments/config/reachability_v1.json`
- `experiments/config/register_bridge_preflight_v1.json`
- `experiments/config/s3_qwen3.json`
- `experiments/config/site_oracle_v1.json`
- `experiments/config/state_bus_v1.json`
- `experiments/config/svb_cross_model.json`
