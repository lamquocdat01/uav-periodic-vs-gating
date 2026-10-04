# uav-periodic-vs-gating

Code, frozen pre-registration and results for the manuscript

> **When Is Periodic Detection Enough for UAV Traffic Monitoring? Theory and a Pre-Registered Test of Cue-Triggered Gating**
> Dat Lam Quoc, FPT School of Business and Technology (FSB), FPT University, Ho Chi Minh City, Vietnam.
> Submitted to *IEEE Transactions on Intelligent Transportation Systems* (under review).

## Summary

An onboard detector on a traffic-monitoring UAV can run on every *S*-th frame (oblivious periodic schedule) or only when a cheap
cue fires (cue-triggered gate). The paper gives a closed theory of both policies for newly appearing vehicles (onset events) — exact
miss and onset-latency expressions, a gain–cost identity whose win set is an interval in the cue's sensitivity, an iso-KPI condition and
a density / ego-motion impossibility result — and tests its predictions in a **pre-registered** experiment on UAVDT: channel parameters
and thresholds were estimated on the TRAIN split, and the predicted sign of the gate advantage was frozen for 72 confirmatory cells
before the TEST split was opened.

Result on UAVDT TEST: **62 correct, 5 wrong, 5 inconclusive** (verdict by the frozen rule: "confirmed";
88.4 % of decided cells correct after removing the 24 iso-KPI cells that no schedule could satisfy). The predicted gate corner did not
appear and no gate beat the periodic schedule at matched cost; criterion (ii) could not be evaluated (no frozen interval for the real channel).

## Pre-registration (verifiable)

| item | value |
|---|---|
| pre-registration document | `PREREG_28.md` — sha256 `81c19b43473e54d9be13be4e7ecaebbaab9e35d479f6027b247afc6b0a3640ab` |
| frozen confirmatory cells | `results/p2/prereg_cells.json` — sha256 `fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e` |
| freeze | tag `prereg-28-v1`, commit `5cfe9014d61faa9e14bc42b7bb1bf3094b752f7c` (2026-10-01 17:04 +07:00) of the development repository |
| TEST opened | once, 2026-10-01, with the frozen scorer (`code/p2/p2_prereg_score.py`) |
| history and deviations | `PREREG_28_DRAFT.md`, `PREREG_28_FREEZE.md`, `LOG_28.md` |

This repository is a clean export of the development repository (which also holds drafts and data-handling notes). The two files
above are byte-identical to the frozen versions (check with `sha256sum`). The only other differences are machine-specific absolute
paths replaced by placeholders, listed with the original hashes in [`RELEASE_CHANGES.md`](RELEASE_CHANGES.md). The frozen
`prereg_cells.json` keeps one absolute path of the development machine in its `channel_source` field, because editing it would change the hash.

## Results: confirmatory family (UAVDT TEST)

| group | cells | correct | wrong | inconclusive |
|---|---|---|---|---|
| H2 | 24 | 24 | 0 | 0 |
| H3-H5 | 24 | 21 | 1 | 2 |
| H8 | 24 | 17 | 4 | 3 |
| **all** | 72 | 62 | 5 | 5 |

Δ̂ = miss_P − miss_G at matched cost (> 0: the gate misses fewer onsets); H8: ΔΔ̂ = Δ̂(yolo26n@640) − Δ̂(yolo26s@1024); H2 (iso-KPI):
a_G + c − a_P(ε) (−∞: no periodic stride reaches ε). 95 % paired sequence-bootstrap CI, B = 1000, seed 42. Full table with counts and
Holm values: `results/p3/cells_test.csv`, `RESULTS_28.md`; post hoc analyses: `RESULTS_28_EXPLORATORY.md`, `results/p3/exploratory/`.

<details><summary>All 72 cells</summary>

| # | hyp. | KPI | ego | M | cue | prediction | TEST estimate | 95 % CI | result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | H2 | iso_L3 | hovering | 6-20 | border band | <=0 | −∞ | [−∞, -0.580] | correct |
| 2 | H2 | iso_L5 | hovering | 6-20 | border band | <=0 | −∞ | [−∞, -0.559] | correct |
| 3 | H2 | iso_L3 | hovering | >20 | border band | <=0 | −∞ | [−∞, −∞] | correct |
| 4 | H2 | iso_L5 | hovering | >20 | border band | <=0 | −∞ | [−∞, −∞] | correct |
| 5 | H2 | iso_L3 | moving | 6-20 | border band | <=0 | −∞ | [−∞, −∞] | correct |
| 6 | H2 | iso_L5 | moving | 6-20 | border band | <=0 | −∞ | [−∞, −∞] | correct |
| 7 | H2 | iso_L3 | hovering | 6-20 | ORB-lite diff. | <=0 | −∞ | [−∞, -0.504] | correct |
| 8 | H2 | iso_L5 | hovering | 6-20 | ORB-lite diff. | <=0 | −∞ | [−∞, -0.446] | correct |
| 9 | H2 | iso_L3 | hovering | >20 | ORB-lite diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 10 | H2 | iso_L5 | hovering | >20 | ORB-lite diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 11 | H2 | iso_L3 | moving | 6-20 | ORB-lite diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 12 | H2 | iso_L5 | moving | 6-20 | ORB-lite diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 13 | H2 | iso_L3 | hovering | 6-20 | raw diff. | <=0 | −∞ | [−∞, -0.520] | correct |
| 14 | H2 | iso_L5 | hovering | 6-20 | raw diff. | <=0 | −∞ | [−∞, -0.458] | correct |
| 15 | H2 | iso_L3 | hovering | >20 | raw diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 16 | H2 | iso_L5 | hovering | >20 | raw diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 17 | H2 | iso_L3 | moving | 6-20 | raw diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 18 | H2 | iso_L5 | moving | 6-20 | raw diff. | <=0 | −∞ | [−∞, −∞] | correct |
| 19 | H2 | iso_L3 | hovering | 6-20 | tiny det. | <=0 | −∞ | [−∞, -0.526] | correct |
| 20 | H2 | iso_L5 | hovering | 6-20 | tiny det. | <=0 | −∞ | [−∞, -0.518] | correct |
| 21 | H2 | iso_L3 | hovering | >20 | tiny det. | <=0 | −∞ | [−∞, −∞] | correct |
| 22 | H2 | iso_L5 | hovering | >20 | tiny det. | <=0 | −∞ | [−∞, −∞] | correct |
| 23 | H2 | iso_L3 | moving | 6-20 | tiny det. | <=0 | −∞ | [−∞, −∞] | correct |
| 24 | H2 | iso_L5 | moving | 6-20 | tiny det. | <=0 | −∞ | [−∞, −∞] | correct |
| 25 | H3-H5 | 3 | hovering | 6-20 | border band | + (+0.040) | -0.099 | [-0.210, +0.220] | inconclusive |
| 26 | H3-H5 | 5 | hovering | 6-20 | border band | <=0 (-0.031) | -0.221 | [-0.322, +0.080] | inconclusive |
| 27 | H3-H5 | 3 | hovering | >20 | border band | + (+0.040) | -0.134 | [-0.193, -0.098] | wrong |
| 28 | H3-H5 | 5 | hovering | >20 | border band | <=0 (-0.030) | -0.209 | [-0.296, -0.158] | correct |
| 29 | H3-H5 | 3 | moving | 6-20 | border band | <=0 (-0.323) | -0.210 | [-0.277, -0.128] | correct |
| 30 | H3-H5 | 5 | moving | 6-20 | border band | <=0 (-0.376) | -0.342 | [-0.421, -0.247] | correct |
| 31 | H3-H5 | 3 | hovering | 6-20 | ORB-lite diff. | <=0 (-0.241) | -0.389 | [-0.519, -0.097] | correct |
| 32 | H3-H5 | 5 | hovering | 6-20 | ORB-lite diff. | <=0 (-0.309) | -0.498 | [-0.635, -0.243] | correct |
| 33 | H3-H5 | 3 | hovering | >20 | ORB-lite diff. | <=0 (-0.245) | -0.393 | [-0.548, -0.303] | correct |
| 34 | H3-H5 | 5 | hovering | >20 | ORB-lite diff. | <=0 (-0.313) | -0.462 | [-0.639, -0.363] | correct |
| 35 | H3-H5 | 3 | moving | 6-20 | ORB-lite diff. | <=0 (-0.437) | -0.513 | [-0.724, -0.286] | correct |
| 36 | H3-H5 | 5 | moving | 6-20 | ORB-lite diff. | <=0 (-0.312) | -0.506 | [-0.706, -0.302] | correct |
| 37 | H3-H5 | 3 | hovering | 6-20 | raw diff. | <=0 (-0.041) | -0.193 | [-0.342, +0.005] | correct |
| 38 | H3-H5 | 5 | hovering | 6-20 | raw diff. | <=0 (-0.129) | -0.391 | [-0.528, -0.209] | correct |
| 39 | H3-H5 | 3 | hovering | >20 | raw diff. | <=0 (-0.043) | -0.326 | [-0.457, -0.250] | correct |
| 40 | H3-H5 | 5 | hovering | >20 | raw diff. | <=0 (-0.131) | -0.458 | [-0.637, -0.356] | correct |
| 41 | H3-H5 | 3 | moving | 6-20 | raw diff. | <=0 (-0.268) | -0.437 | [-0.750, -0.086] | correct |
| 42 | H3-H5 | 5 | moving | 6-20 | raw diff. | <=0 (-0.144) | -0.431 | [-0.723, -0.111] | correct |
| 43 | H3-H5 | 3 | hovering | 6-20 | tiny det. | <=0 (-0.431) | -0.313 | [-0.515, -0.147] | correct |
| 44 | H3-H5 | 5 | hovering | 6-20 | tiny det. | <=0 (-0.388) | -0.361 | [-0.538, -0.179] | correct |
| 45 | H3-H5 | 3 | hovering | >20 | tiny det. | <=0 (-0.507) | -0.444 | [-0.636, -0.333] | correct |
| 46 | H3-H5 | 5 | hovering | >20 | tiny det. | <=0 (-0.391) | -0.509 | [-0.718, -0.387] | correct |
| 47 | H3-H5 | 3 | moving | 6-20 | tiny det. | <=0 (-0.375) | -0.575 | [-0.719, -0.311] | correct |
| 48 | H3-H5 | 5 | moving | 6-20 | tiny det. | <=0 (-0.396) | -0.564 | [-0.676, -0.302] | correct |
| 49 | H8 | H8_L3 | hovering | 6-20 | border band | <=0 (+0.002) | +0.024 | [+0.006, +0.032] | wrong |
| 50 | H8 | H8_L5 | hovering | 6-20 | border band | + (+0.011) | +0.043 | [+0.009, +0.062] | correct |
| 51 | H8 | H8_L3 | hovering | >20 | border band | <=0 (+0.002) | +0.033 | [+0.015, +0.069] | wrong |
| 52 | H8 | H8_L5 | hovering | >20 | border band | + (+0.011) | +0.049 | [+0.020, +0.107] | correct |
| 53 | H8 | H8_L3 | moving | 6-20 | border band | + (+0.028) | +0.047 | [+0.006, +0.082] | correct |
| 54 | H8 | H8_L5 | moving | 6-20 | border band | + (+0.028) | +0.078 | [+0.009, +0.125] | correct |
| 55 | H8 | H8_L3 | hovering | 6-20 | ORB-lite diff. | + (+0.025) | +0.073 | [+0.011, +0.125] | correct |
| 56 | H8 | H8_L5 | hovering | 6-20 | ORB-lite diff. | + (+0.025) | +0.078 | [+0.003, +0.143] | correct |
| 57 | H8 | H8_L3 | hovering | >20 | ORB-lite diff. | + (+0.026) | +0.098 | [+0.039, +0.228] | correct |
| 58 | H8 | H8_L5 | hovering | >20 | ORB-lite diff. | + (+0.026) | +0.107 | [+0.032, +0.254] | correct |
| 59 | H8 | H8_L3 | moving | 6-20 | ORB-lite diff. | + (+0.030) | +0.130 | [+0.020, +0.223] | correct |
| 60 | H8 | H8_L5 | moving | 6-20 | ORB-lite diff. | <=0 (-0.004) | +0.114 | [+0.016, +0.199] | wrong |
| 61 | H8 | H8_L3 | hovering | 6-20 | raw diff. | + (+0.021) | +0.046 | [+0.009, +0.069] | correct |
| 62 | H8 | H8_L5 | hovering | 6-20 | raw diff. | + (+0.029) | +0.073 | [+0.011, +0.104] | correct |
| 63 | H8 | H8_L3 | hovering | >20 | raw diff. | + (+0.021) | +0.082 | [+0.037, +0.186] | correct |
| 64 | H8 | H8_L5 | hovering | >20 | raw diff. | + (+0.029) | +0.106 | [+0.036, +0.244] | correct |
| 65 | H8 | H8_L3 | moving | 6-20 | raw diff. | + (+0.008) | +0.107 | [+0.008, +0.214] | correct |
| 66 | H8 | H8_L5 | moving | 6-20 | raw diff. | <=0 (-0.016) | +0.089 | [+0.001, +0.181] | inconclusive |
| 67 | H8 | H8_L3 | hovering | 6-20 | tiny det. | + (+0.037) | +0.051 | [-0.014, +0.100] | inconclusive |
| 68 | H8 | H8_L5 | hovering | 6-20 | tiny det. | + (+0.016) | +0.053 | [-0.021, +0.096] | inconclusive |
| 69 | H8 | H8_L3 | hovering | >20 | tiny det. | + (+0.034) | +0.106 | [+0.047, +0.241] | correct |
| 70 | H8 | H8_L5 | hovering | >20 | tiny det. | <=0 (+0.000) | +0.114 | [+0.044, +0.258] | wrong |
| 71 | H8 | H8_L3 | moving | 6-20 | tiny det. | + (+0.033) | +0.129 | [+0.025, +0.210] | correct |
| 72 | H8 | H8_L5 | moving | 6-20 | tiny det. | + (+0.022) | +0.104 | [+0.011, +0.180] | correct |

</details>

## Repository structure

```
PREREG_28*.md            pre-registration (frozen document, draft with criteria wording, freeze manifest)
THEORY_28.md             theory (source of Sections III-IV and Supplementary S1)
RESULTS_28.md            confirmatory results (generated by code/p3/p3_score_report.py)
RESULTS_28_EXPLORATORY.md  post hoc analyses (generated; not part of the confirmatory family)
LOG_28.md                project log (decisions, deviations, timings)
DOSSIER_TITS_28.md       literature search protocol behind the "to our knowledge" statements (Vietnamese)
ENV_28.md, DATA_SOURCES.md  environment and data sources
code/paths.py            all data paths (dataset root from $THS_DATASETS)
code/p0_*.py             UAVDT parsing, onset events, ego-motion, premise statistics
code/theory_checks.py    numerical checks of every theorem (results/p1/theory_checks.json)
code/p2/                 frames, detector dumps, cues, channel estimation, pre-registration build and the frozen scorer
code/p2/replay/          offline replay engine (periodic / gate-or-refresh), simulator used to test the pipeline
code/p3/                 TEST preparation, replay, scoring report; exploratory/ = post hoc (P3b)
code/p4/                 numbers.tex / tables and figures of the paper
code/kaggle/             Kaggle notebooks used to train YOLO26-s/-n on VisDrone2019-DET
tests/                   pytest-style tests (fixtures are synthesised by the tests)
results/p1, p2, p3       small derived results (JSON/CSV); results/p2/sim = simulated-channel checks
paper/figs, paper/figures  figures of the paper (PDF/PNG); paper/numbers.tex, paper/tables = generated macros
```

## Reproduction

Environment: Python 3.11.9 (Windows 11, Intel Core i7-1185G7, Iris Xe iGPU); `pip install -r requirements-lock.txt`
(ultralytics 8.4.163, openvino 2026.4.0). Datasets are **not** redistributed: download UAVDT (and VisDrone2019-DET for training)
from the official sources and set `THS_DATASETS` to their parent folder (layout in `code/paths.py`, sources in `DATA_SOURCES.md`).

Detector weights are not included (Kaggle training outputs; see `code/kaggle/README_KAGGLE_28.md`):

| detector | Kaggle output | sha256 of `weights/best.pt` |
|---|---|---|
| YOLO26-s, VisDrone2019-DET, 1024 (main detector) | <!-- TODO-DAT: public Kaggle output URL --> to be added | `825dffcc41f8bfcd871d97b1e48b8fc3ba8fad2002cd3c0e743ae07e5a57bf84` |
| YOLO26-n, VisDrone2019-DET, 1024 (low-recall level @640, tiny cue @320) | <!-- TODO-DAT: public Kaggle output URL --> to be added | `4dc79ab7b900ee4ba382719e3c101ce8a56b6f045c3bbc9ae0af15d3b29472e3` |

From detector dumps to the score (one detector worker at a time; heavy jobs stop below 6 GB free RAM):

```bash
python code/p0_parse.py && python code/p0_events.py && python code/p0_egomotion.py      # events, ego-motion (TRAIN)
python code/p2/p2_frames.py                                                              # frame index
python code/p2/p2_dump.py --detector yolo26s --imgsz 1024 --split all --device GPU --max-workers 1 --min-free-ram-gb 6
python code/p2/p2_dump.py --detector yolo26n --imgsz 640  --split all --device GPU --max-workers 1 --min-free-ram-gb 6
python code/p2/p2_cue_trace.py --dataset uavdt --split all
python code/p2/p2_detector_recall.py                                                     # per-look recall (TRAIN)
python code/p2/p2_estimate_channel.py --dataset uavdt                                    # TRAIN channel
python code/p2/p2_prereg_build.py                                                        # -> results/p2/prereg_cells.json
python code/p3/p3_prepare_test.py                                                        # TEST events, ego-motion with frozen threshold
python code/p3/p3_replay_test.py --split test                                            # -> results/p3/obs_test.json
python code/p2/p2_prereg_score.py --cells results/p2/prereg_cells.json --obs results/p3/obs_test.json
python code/p3/p3_score_report.py                                                        # -> RESULTS_28.md
python code/p3/exploratory/p3b_explore.py                                                # post hoc (P3b)
python code/p4/p4_numbers.py && python code/p4/p4_figures.py                             # numbers.tex, tables, figures
```

Rebuilding `prereg_cells.json` reproduces the frozen file only with the same dumps and cue traces; to check the scoring alone, run
the last five commands on the released `results/p3/obs_test.json`. Run the tests with `python -m pytest tests`.

## Licence

Code (`code/`, `tests/`): MIT ([`LICENSE-MIT`](LICENSE-MIT)). Documents, results and figures: CC BY 4.0 ([`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0)).
UAVDT and VisDrone remain under their own licences.

## Citation

See [`CITATION.cff`](CITATION.cff). Release used for the submission: `v1.0-submission`.
