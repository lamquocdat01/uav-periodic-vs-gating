# RESULTS_28 — UAVDT TEST, họ xác nhận đóng băng (sinh tự động)

*Sinh bởi `code/p3/p3_score_report.py` — 2026-10-03T16:19:10. PREREG: tag `prereg-28-v1`; prereg_cells.json sha256 `fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e` (= FREEZE). Quan sát: `code/p3/p3_replay_test.py` → results/p3/obs_test.json. Chấm: `code/p2/p2_prereg_score.py` (đóng băng). File này không chứa diễn giải ngoài câu chữ tiêu chí.*

## Tổng hợp 3 mức

| nhóm | n | ĐÚNG | SAI | CHƯA KẾT LUẬN |
|---|---|---|---|---|
| H2 | 24 | 24 | 0 | 0 |
| H3-H5 | 24 | 21 | 1 | 2 |
| H8 | 24 | 17 | 4 | 3 |
| **tổng** | 72 | 62 | 5 | 5 |

- Ô dự đoán "+": 21 — ĐÚNG 17, SAI 1, CHƯA KẾT LUẬN 3.
- Ô không đạt luật cỡ mẫu trên TEST (≥ 30 sự kiện, ≥ 3 chuỗi) → CHƯA KẾT LUẬN: 0.
- Holm (chỉ báo cáo): 22 ô có ý nghĩa sau hiệu chỉnh.

## Tiêu chí (câu chữ PREREG_28_DRAFT §4)

| tiêu chí | câu chữ | giá trị TEST | đạt điều kiện |
|---|---|---|---|
| P | If more than 50 % of the confirmatory cells are INCONCLUSIVE, the paper's conclusion is "not testable with these data" (neither confirmed nor rejected). | CHƯA KẾT LUẬN 6.9 % | không |
| 1 | fewer than 80 % of the confirmatory cells that are DECIDED (correct or wrong) are correct | ĐÚNG / KẾT LUẬN ĐƯỢC = 92.5 % (67 ô kết luận được) | không |
| 2 | the observed iso-KPI boundary q_o* (THEORY Thm P3-iii; P2B-A1) lies outside its predicted bootstrap CI in ≥ 2/3 of the density groups | không đánh giá: không có CI dự đoán q_o* đóng băng cho kênh thật trong prereg_cells.json → tiêu chí (ii) không đánh giá | — |
| 3 | the sign of H8 ("Δ(latency) increases as recall decreases") is wrong in ≥ 50 % of detector pairs (evaluated on decided H8 cells) | SAI / H8 kết luận được = 19.0 % | không |

**Kết luận (theo câu chữ tiêu chí): confirmed.**

## Bảng 72 ô

Δ̂ = miss_P − miss_G ở matched cost (> 0: gate thắng); H8: ΔΔ̂ = Δ̂(yolo26n_640) − Δ̂(yolo26s_1024); H2 (iso-KPI): a_G − a_P(ε). CI 95 % bootstrap ghép cặp theo chuỗi, B = 1000, seed 42.

| nhóm | KPI | ε | ego | M_bin | cue | dự đoán | giá trị dự đoán | quan sát | CI 95 % | gate rẻ hơn (H2) | kết quả | n sự kiện / chuỗi TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2 | iso_L3 | 0.02 | hovering | 6-20 | border_band | <=0 | a_G 0.099 / a_P 0.483 | −∞ | [−∞; -0.5805] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L5 | 0.02 | hovering | 6-20 | border_band | <=0 | a_G 0.099 / a_P 0.382 | −∞ | [−∞; -0.5591] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L3 | 0.02 | hovering | >20 | border_band | <=0 | a_G 0.099 / a_P 0.457 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L5 | 0.02 | hovering | >20 | border_band | <=0 | a_G 0.099 / a_P 0.316 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L3 | 0.02 | moving | 6-20 | border_band | <=0 | a_G 0.196 / a_P 0.452 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L5 | 0.02 | moving | 6-20 | border_band | <=0 | a_G 0.196 / a_P 0.307 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L3 | 0.02 | hovering | 6-20 | orb_lite_diff | <=0 | a_G 0.201 / a_P 0.483 | −∞ | [−∞; -0.5037] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L5 | 0.02 | hovering | 6-20 | orb_lite_diff | <=0 | a_G 0.201 / a_P 0.382 | −∞ | [−∞; -0.4462] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L3 | 0.02 | hovering | >20 | orb_lite_diff | <=0 | a_G 0.201 / a_P 0.457 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L5 | 0.02 | hovering | >20 | orb_lite_diff | <=0 | a_G 0.201 / a_P 0.316 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L3 | 0.02 | moving | 6-20 | orb_lite_diff | <=0 | a_G 0.292 / a_P 0.452 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L5 | 0.02 | moving | 6-20 | orb_lite_diff | <=0 | a_G 0.292 / a_P 0.307 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L3 | 0.02 | hovering | 6-20 | raw_diff | <=0 | a_G 0.197 / a_P 0.483 | −∞ | [−∞; -0.5199] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L5 | 0.02 | hovering | 6-20 | raw_diff | <=0 | a_G 0.197 / a_P 0.382 | −∞ | [−∞; -0.4578] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L3 | 0.02 | hovering | >20 | raw_diff | <=0 | a_G 0.197 / a_P 0.457 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L5 | 0.02 | hovering | >20 | raw_diff | <=0 | a_G 0.197 / a_P 0.316 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L3 | 0.02 | moving | 6-20 | raw_diff | <=0 | a_G 0.377 / a_P 0.452 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L5 | 0.02 | moving | 6-20 | raw_diff | <=0 | a_G 0.377 / a_P 0.307 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L3 | 0.02 | hovering | 6-20 | tiny_det | <=0 | a_G 0.233 / a_P 0.483 | −∞ | [−∞; -0.5257] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L5 | 0.02 | hovering | 6-20 | tiny_det | <=0 | a_G 0.233 / a_P 0.382 | −∞ | [−∞; -0.5182] | không | ĐÚNG | 121 / 8 |
| H2 | iso_L3 | 0.02 | hovering | >20 | tiny_det | <=0 | a_G 0.284 / a_P 0.457 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L5 | 0.02 | hovering | >20 | tiny_det | <=0 | a_G 0.284 / a_P 0.316 | −∞ | [−∞; −∞] | không | ĐÚNG | 332 / 9 |
| H2 | iso_L3 | 0.02 | moving | 6-20 | tiny_det | <=0 | a_G 0.217 / a_P 0.452 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H2 | iso_L5 | 0.02 | moving | 6-20 | tiny_det | <=0 | a_G 0.217 / a_P 0.307 | −∞ | [−∞; −∞] | không | ĐÚNG | 221 / 7 |
| H3-H5 | 3 | — | hovering | 6-20 | border_band | + | +0.0401 | -0.0988 | [-0.2100; +0.2198] | — | CHƯA KẾT LUẬN | 121 / 8 |
| H3-H5 | 5 | — | hovering | 6-20 | border_band | <=0 | -0.0306 | -0.2211 | [-0.3220; +0.0802] | — | CHƯA KẾT LUẬN | 121 / 8 |
| H3-H5 | 3 | — | hovering | >20 | border_band | + | +0.0402 | -0.1342 | [-0.1935; -0.0978] | — | SAI | 332 / 9 |
| H3-H5 | 5 | — | hovering | >20 | border_band | <=0 | -0.0301 | -0.2093 | [-0.2959; -0.1580] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 3 | — | moving | 6-20 | border_band | <=0 | -0.3231 | -0.2102 | [-0.2770; -0.1280] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 5 | — | moving | 6-20 | border_band | <=0 | -0.3760 | -0.3423 | [-0.4215; -0.2466] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 3 | — | hovering | 6-20 | orb_lite_diff | <=0 | -0.2413 | -0.3888 | [-0.5188; -0.0965] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 5 | — | hovering | 6-20 | orb_lite_diff | <=0 | -0.3094 | -0.4979 | [-0.6350; -0.2428] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 3 | — | hovering | >20 | orb_lite_diff | <=0 | -0.2448 | -0.3931 | [-0.5480; -0.3034] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 5 | — | hovering | >20 | orb_lite_diff | <=0 | -0.3131 | -0.4623 | [-0.6390; -0.3629] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 3 | — | moving | 6-20 | orb_lite_diff | <=0 | -0.4366 | -0.5129 | [-0.7241; -0.2864] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 5 | — | moving | 6-20 | orb_lite_diff | <=0 | -0.3116 | -0.5063 | [-0.7062; -0.3018] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 3 | — | hovering | 6-20 | raw_diff | <=0 | -0.0407 | -0.1926 | [-0.3420; +0.0049] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 5 | — | hovering | 6-20 | raw_diff | <=0 | -0.1291 | -0.3909 | [-0.5278; -0.2094] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 3 | — | hovering | >20 | raw_diff | <=0 | -0.0428 | -0.3256 | [-0.4571; -0.2496] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 5 | — | hovering | >20 | raw_diff | <=0 | -0.1313 | -0.4578 | [-0.6365; -0.3559] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 3 | — | moving | 6-20 | raw_diff | <=0 | -0.2682 | -0.4371 | [-0.7496; -0.0856] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 5 | — | moving | 6-20 | raw_diff | <=0 | -0.1438 | -0.4310 | [-0.7228; -0.1105] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 3 | — | hovering | 6-20 | tiny_det | <=0 | -0.4310 | -0.3132 | [-0.5150; -0.1475] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 5 | — | hovering | 6-20 | tiny_det | <=0 | -0.3885 | -0.3612 | [-0.5380; -0.1795] | — | ĐÚNG | 121 / 8 |
| H3-H5 | 3 | — | hovering | >20 | tiny_det | <=0 | -0.5065 | -0.4444 | [-0.6357; -0.3328] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 5 | — | hovering | >20 | tiny_det | <=0 | -0.3908 | -0.5086 | [-0.7176; -0.3870] | — | ĐÚNG | 332 / 9 |
| H3-H5 | 3 | — | moving | 6-20 | tiny_det | <=0 | -0.3752 | -0.5751 | [-0.7189; -0.3111] | — | ĐÚNG | 221 / 7 |
| H3-H5 | 5 | — | moving | 6-20 | tiny_det | <=0 | -0.3962 | -0.5640 | [-0.6764; -0.3021] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L3 | — | hovering | 6-20 | border_band | <=0 | +0.0020 | +0.0236 | [+0.0056; +0.0322] | — | SAI | 121 / 8 |
| H8 | H8_L5 | — | hovering | 6-20 | border_band | + | +0.0112 | +0.0430 | [+0.0087; +0.0617] | — | ĐÚNG | 121 / 8 |
| H8 | H8_L3 | — | hovering | >20 | border_band | <=0 | +0.0020 | +0.0325 | [+0.0148; +0.0694] | — | SAI | 332 / 9 |
| H8 | H8_L5 | — | hovering | >20 | border_band | + | +0.0112 | +0.0494 | [+0.0200; +0.1072] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L3 | — | moving | 6-20 | border_band | + | +0.0281 | +0.0473 | [+0.0062; +0.0824] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L5 | — | moving | 6-20 | border_band | + | +0.0282 | +0.0778 | [+0.0090; +0.1252] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L3 | — | hovering | 6-20 | orb_lite_diff | + | +0.0254 | +0.0731 | [+0.0111; +0.1254] | — | ĐÚNG | 121 / 8 |
| H8 | H8_L5 | — | hovering | 6-20 | orb_lite_diff | + | +0.0251 | +0.0781 | [+0.0032; +0.1434] | — | ĐÚNG | 121 / 8 |
| H8 | H8_L3 | — | hovering | >20 | orb_lite_diff | + | +0.0258 | +0.0979 | [+0.0386; +0.2278] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L5 | — | hovering | >20 | orb_lite_diff | + | +0.0256 | +0.1068 | [+0.0315; +0.2537] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L3 | — | moving | 6-20 | orb_lite_diff | + | +0.0298 | +0.1299 | [+0.0201; +0.2227] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L5 | — | moving | 6-20 | orb_lite_diff | <=0 | -0.0042 | +0.1136 | [+0.0156; +0.1990] | — | SAI | 221 / 7 |
| H8 | H8_L3 | — | hovering | 6-20 | raw_diff | + | +0.0210 | +0.0459 | [+0.0091; +0.0688] | — | ĐÚNG | 121 / 8 |
| H8 | H8_L5 | — | hovering | 6-20 | raw_diff | + | +0.0285 | +0.0727 | [+0.0114; +0.1039] | — | ĐÚNG | 121 / 8 |
| H8 | H8_L3 | — | hovering | >20 | raw_diff | + | +0.0214 | +0.0824 | [+0.0367; +0.1859] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L5 | — | hovering | >20 | raw_diff | + | +0.0290 | +0.1060 | [+0.0362; +0.2440] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L3 | — | moving | 6-20 | raw_diff | + | +0.0080 | +0.1070 | [+0.0085; +0.2142] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L5 | — | moving | 6-20 | raw_diff | <=0 | -0.0160 | +0.0891 | [+0.0014; +0.1807] | — | CHƯA KẾT LUẬN | 221 / 7 |
| H8 | H8_L3 | — | hovering | 6-20 | tiny_det | + | +0.0365 | +0.0512 | [-0.0139; +0.0997] | — | CHƯA KẾT LUẬN | 121 / 8 |
| H8 | H8_L5 | — | hovering | 6-20 | tiny_det | + | +0.0162 | +0.0533 | [-0.0215; +0.0957] | — | CHƯA KẾT LUẬN | 121 / 8 |
| H8 | H8_L3 | — | hovering | >20 | tiny_det | + | +0.0342 | +0.1065 | [+0.0473; +0.2413] | — | ĐÚNG | 332 / 9 |
| H8 | H8_L5 | — | hovering | >20 | tiny_det | <=0 | +0.0003 | +0.1136 | [+0.0436; +0.2582] | — | SAI | 332 / 9 |
| H8 | H8_L3 | — | moving | 6-20 | tiny_det | + | +0.0326 | +0.1285 | [+0.0248; +0.2096] | — | ĐÚNG | 221 / 7 |
| H8 | H8_L5 | — | moving | 6-20 | tiny_det | + | +0.0218 | +0.1036 | [+0.0112; +0.1796] | — | ĐÚNG | 221 / 7 |

Chi tiết: results/p3/cells_test.csv, results/p3/score_test.json.
