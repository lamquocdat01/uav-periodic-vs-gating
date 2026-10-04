# RELEASE_CHANGES — differences between this export and the development repository

This repository is a clean export of the development repository of topic 28 (pre-registration freeze: tag `prereg-28-v1`, commit `5cfe901`). The only edits are machine-specific absolute paths, replaced by the placeholders `<repo>`, `<workspace>`, `$THS_DATASETS` and `<home>` (and the default of `THS_DATASETS` in `code/paths.py`). Byte-frozen files whose hashes are cited in the paper are copied unchanged: `PREREG_28.md`, `results/p2/prereg_cells.json`.

| file | sha256 of the original (development repository) |
|---|---|
| `LOG_28.md` | `49c66da1ccb3a0b199f00311491a434c71e5978e045654315785221de1cfd30d` |
| `ENV_28.md` | `3bc2e090be7fcba344e421f61f8393754c37214fb8baa38780b3c599b38e8c36` |
| `DATA_SOURCES.md` | `94917219b44f474d38ab007fc085434f445e58d7e13333a13cd5c1ddd45fed18` |
| `code/p0_premise.py` | `0a18a577758421726ef990f0be9f11d82b3ef6fab0fd6276cee9388326d15ffa` |
| `code/paths.py` | `218eb6dbce3394aa66103decf42db986e3019136b1fe70b4e9f1e82cc533f5ee` |
| `code/retry_uavdt.sh` | `9eb74dc4c9b734e7d80a8f0dd3ed8a87cfa79302b7631ec41308b6f746733604` |
| `code/retry_visdrone.sh` | `43e83505e67a9fafed4e0a29861d4f9707adc8b03d83f70fbab0be109fcb1d65` |
| `code/p2/p2_dump_queue.ps1` | `3e78e45b89945df0d33370f88cad4477499948fd58edd1bdbcd33fc805e2c4b3` |
| `code/p2/p2_dump_queue_hidden.ps1` | `3a56826ea90f741539fa980e265c205855a9aaba2d35bfca40b793d317e4b2b4` |
| `code/p2/p2_wait_idle_bench.ps1` | `ae5a5801f7b560982ba47fa28ab262a6aeb2e0450000991cf5f55e7887a67d8b` |
| `code/kaggle/README_KAGGLE_28.md` | `a80cb1e0531f2ba1f5c1aa94c68b0797ff40d195069db8fb4e60fa601c9299ba` |
| `results/p2/bench_openvino.json` | `12e4dc098d4fba2746709a0878f18496de655380873ff2e07f988c39a1b18994` |
| `results/p2/dump_queue_status.json` | `e28e88a52691d64ff830027dca43992b45911b55bff0fd1ac2869801e0f9ef18` |
| `results/p2/env_p2.json` | `815d3e4f54597b887cbc180091ae0ef0f52c60a6aef18f949909dc2befef5f0f` |
| `results/p2/uavdt_frames_verify.json` | `12c5ed619d2bd27648a8dac201f7beeb3f8261b7818f55e0e49eb88ca2979ee5` |
| `results/p2/verify_weights_v2.json` | `08624ede56c59628beac05c7d6b07deb402c88a98d2d2a42ce3816cc1ea0966f` |
| `results/p2/verify_yolo26n_visdrone_1024.json` | `4bf167db848e3ae83cff0ee33784168b1107ab1444b5e842319c286283930392` |
| `results/p2/verify_yolo26s_visdrone_1024.json` | `8663c5a7f0147631f2fc4385fa067352c861913d0521d654326eff9b90eba01f` |
| `results/p2/verify_yolo26s_visdrone_1024_nonms.json` | `c795bfba77a94dc7e5da7ee2f2d19444d3107d3002284c14d847c7a30f39fdfd` |
