# RESULTS_28_EXPLORATORY — phân tích hậu kiểm sau khi mở TEST (sinh tự động)

*Sinh bởi `code/p3/exploratory/p3b_report.py`. Mọi mục dưới đây là [POST HOC — không thuộc họ xác nhận]: không đổi nhãn, tiêu chí hay kết luận của RESULTS_28.md (họ xác nhận đóng băng). Số liệu: results/p3/exploratory/*.json.*

## [POST HOC — không thuộc họ xác nhận] E1 — độ khả thi iso-KPI (vì sao 24 ô H2 đúng một cách tầm thường)

| tầng | L | n sự kiện | sàn periodic miss_P(S = 1) = ε_feasible | sàn gate min_θ miss_G (cue, θ) | a_P(2 %) dự đoán PREREG (theo cue) | a_P(2 %) quan sát |
|---|---|---|---|---|---|---|
| hovering 6-20 | 3 | 121 | 5.0 % | 12.9 % (tiny_det, θ 0.782) | border_band 0.483, orb_lite_diff 0.483, raw_diff 0.483, tiny_det 0.483 | ∞ |
| hovering 6-20 | 5 | 121 | 4.1 % | 11.6 % (orb_lite_diff, θ 0.008746) | border_band 0.382, orb_lite_diff 0.382, raw_diff 0.382, tiny_det 0.382 | ∞ |
| hovering >20 | 3 | 332 | 32.2 % | 59.6 % (orb_lite_diff, θ 0.008746) | border_band 0.457, orb_lite_diff 0.457, raw_diff 0.457, tiny_det 0.457 | ∞ |
| hovering >20 | 5 | 332 | 30.7 % | 54.7 % (orb_lite_diff, θ 0.008746) | border_band 0.316, orb_lite_diff 0.316, raw_diff 0.316, tiny_det 0.316 | ∞ |
| moving 6-20 | 3 | 221 | 11.3 % | 24.7 % (border_band, θ 0.008786) | border_band 0.452, orb_lite_diff 0.452, raw_diff 0.452, tiny_det 0.452 | ∞ |
| moving 6-20 | 5 | 221 | 8.6 % | 20.2 % (border_band, θ 0.008786) | border_band 0.307, orb_lite_diff 0.307, raw_diff 0.307, tiny_det 0.307 | ∞ |

Iso-KPI ở ε nới (θ* đóng băng, R = 30; CI bootstrap theo chuỗi B = 1000 seed 42; **không chấm điểm**):

| tầng | L | ε | cue | miss_G | gate đạt ε | a_G | a_P(ε) | a_G − a_P | CI 95 % | dấu |
|---|---|---|---|---|---|---|---|---|---|---|
| hovering 6-20 | 3 | 7.0 % (feasible+2pt) | border_band | 0.748 | không (0.0 % mẫu boot) | 0.096 | 0.417 | -0.320 | [—; -0.121] | − |
| hovering 6-20 | 3 | 7.0 % (feasible+2pt) | orb_lite_diff | 0.647 | không (0.0 % mẫu boot) | 0.205 | 0.417 | -0.211 | [—; +0.000] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 7.0 % (feasible+2pt) | raw_diff | 0.538 | không (0.0 % mẫu boot) | 0.182 | 0.417 | -0.235 | [—; -0.008] | − |
| hovering 6-20 | 3 | 7.0 % (feasible+2pt) | tiny_det | 0.540 | không (0.0 % mẫu boot) | 0.213 | 0.417 | -0.203 | [—; -0.003] | − |
| hovering 6-20 | 3 | 10.0 % (fixed) | border_band | 0.748 | không (0.0 % mẫu boot) | 0.096 | 0.248 | -0.152 | [—; -0.111] | − |
| hovering 6-20 | 3 | 10.0 % (fixed) | orb_lite_diff | 0.647 | không (0.0 % mẫu boot) | 0.205 | 0.248 | -0.043 | [—; +0.019] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 10.0 % (fixed) | raw_diff | 0.538 | không (0.0 % mẫu boot) | 0.182 | 0.248 | -0.067 | [—; +0.004] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 10.0 % (fixed) | tiny_det | 0.540 | không (0.0 % mẫu boot) | 0.213 | 0.248 | -0.035 | [—; +0.023] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 15.0 % (fixed) | border_band | 0.748 | không (0.0 % mẫu boot) | 0.096 | 0.235 | -0.138 | [-0.347; -0.097] | − |
| hovering 6-20 | 3 | 15.0 % (fixed) | orb_lite_diff | 0.647 | không (0.0 % mẫu boot) | 0.205 | 0.235 | -0.029 | [-0.243; +0.035] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 15.0 % (fixed) | raw_diff | 0.538 | không (0.0 % mẫu boot) | 0.182 | 0.235 | -0.053 | [-0.289; +0.019] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 15.0 % (fixed) | tiny_det | 0.540 | không (0.0 % mẫu boot) | 0.213 | 0.235 | -0.021 | [-0.237; +0.039] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 20.0 % (fixed) | border_band | 0.748 | không (0.0 % mẫu boot) | 0.096 | 0.221 | -0.124 | [-0.167; -0.084] | − |
| hovering 6-20 | 3 | 20.0 % (fixed) | orb_lite_diff | 0.647 | không (0.0 % mẫu boot) | 0.205 | 0.221 | -0.015 | [-0.079; +0.048] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 20.0 % (fixed) | raw_diff | 0.538 | không (0.0 % mẫu boot) | 0.182 | 0.221 | -0.039 | [-0.132; +0.031] | 0 (CI chứa 0) |
| hovering 6-20 | 3 | 20.0 % (fixed) | tiny_det | 0.540 | không (0.0 % mẫu boot) | 0.213 | 0.221 | -0.007 | [-0.068; +0.052] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 6.1 % (feasible+2pt) | border_band | 0.697 | không (0.0 % mẫu boot) | 0.096 | 0.331 | -0.235 | [—; -0.046] | − |
| hovering 6-20 | 5 | 6.1 % (feasible+2pt) | orb_lite_diff | 0.583 | không (0.0 % mẫu boot) | 0.205 | 0.331 | -0.126 | [—; +0.068] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 6.1 % (feasible+2pt) | raw_diff | 0.481 | không (0.0 % mẫu boot) | 0.182 | 0.331 | -0.150 | [—; +0.062] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 6.1 % (feasible+2pt) | tiny_det | 0.445 | không (0.0 % mẫu boot) | 0.213 | 0.331 | -0.118 | [—; +0.058] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 10.0 % (fixed) | border_band | 0.697 | không (0.0 % mẫu boot) | 0.096 | 0.166 | -0.069 | [-0.745; -0.031] | − |
| hovering 6-20 | 5 | 10.0 % (fixed) | orb_lite_diff | 0.583 | không (0.0 % mẫu boot) | 0.205 | 0.166 | +0.040 | [-0.671; +0.097] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 10.0 % (fixed) | raw_diff | 0.481 | không (0.0 % mẫu boot) | 0.182 | 0.166 | +0.016 | [-0.717; +0.083] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 10.0 % (fixed) | tiny_det | 0.445 | không (0.0 % mẫu boot) | 0.213 | 0.166 | +0.047 | [-0.659; +0.104] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 15.0 % (fixed) | border_band | 0.697 | không (0.0 % mẫu boot) | 0.096 | 0.156 | -0.060 | [-0.199; -0.022] | − |
| hovering 6-20 | 5 | 15.0 % (fixed) | orb_lite_diff | 0.583 | không (0.0 % mẫu boot) | 0.205 | 0.156 | +0.049 | [-0.106; +0.110] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 15.0 % (fixed) | raw_diff | 0.481 | không (0.0 % mẫu boot) | 0.182 | 0.156 | +0.025 | [-0.157; +0.093] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 15.0 % (fixed) | tiny_det | 0.445 | không (0.0 % mẫu boot) | 0.213 | 0.156 | +0.057 | [-0.105; +0.115] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 20.0 % (fixed) | border_band | 0.697 | không (0.0 % mẫu boot) | 0.096 | 0.148 | -0.051 | [-0.086; -0.013] | − |
| hovering 6-20 | 5 | 20.0 % (fixed) | orb_lite_diff | 0.583 | không (0.0 % mẫu boot) | 0.205 | 0.148 | +0.058 | [+0.001; +0.120] | + |
| hovering 6-20 | 5 | 20.0 % (fixed) | raw_diff | 0.481 | không (0.1 % mẫu boot) | 0.182 | 0.148 | +0.034 | [-0.049; +0.102] | 0 (CI chứa 0) |
| hovering 6-20 | 5 | 20.0 % (fixed) | tiny_det | 0.445 | không (0.2 % mẫu boot) | 0.213 | 0.148 | +0.066 | [+0.013; +0.124] | + |
| hovering >20 | 3 | 34.2 % (feasible+2pt) | border_band | 0.893 | không (0.0 % mẫu boot) | 0.096 | 0.586 | -0.490 | [—; -0.090] | − |
| hovering >20 | 3 | 34.2 % (feasible+2pt) | orb_lite_diff | 0.880 | không (0.0 % mẫu boot) | 0.205 | 0.586 | -0.381 | [—; +0.045] | 0 (CI chứa 0) |
| hovering >20 | 3 | 34.2 % (feasible+2pt) | raw_diff | 0.875 | không (0.0 % mẫu boot) | 0.182 | 0.586 | -0.405 | [—; +0.030] | 0 (CI chứa 0) |
| hovering >20 | 3 | 34.2 % (feasible+2pt) | tiny_det | 0.914 | không (0.0 % mẫu boot) | 0.213 | 0.586 | -0.373 | [—; +0.054] | 0 (CI chứa 0) |
| hovering >20 | 3 | 10.0 % (fixed) | border_band | 0.893 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.199] | − |
| hovering >20 | 3 | 10.0 % (fixed) | orb_lite_diff | 0.880 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; -0.092] | − |
| hovering >20 | 3 | 10.0 % (fixed) | raw_diff | 0.875 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; -0.108] | − |
| hovering >20 | 3 | 10.0 % (fixed) | tiny_det | 0.914 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; -0.064] | − |
| hovering >20 | 3 | 15.0 % (fixed) | border_band | 0.893 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.148] | − |
| hovering >20 | 3 | 15.0 % (fixed) | orb_lite_diff | 0.880 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; -0.027] | − |
| hovering >20 | 3 | 15.0 % (fixed) | raw_diff | 0.875 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; -0.035] | − |
| hovering >20 | 3 | 15.0 % (fixed) | tiny_det | 0.914 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; -0.010] | − |
| hovering >20 | 3 | 20.0 % (fixed) | border_band | 0.893 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.132] | − |
| hovering >20 | 3 | 20.0 % (fixed) | orb_lite_diff | 0.880 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; -0.007] | − |
| hovering >20 | 3 | 20.0 % (fixed) | raw_diff | 0.875 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; -0.016] | − |
| hovering >20 | 3 | 20.0 % (fixed) | tiny_det | 0.914 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; +0.011] | 0 (CI chứa 0) |
| hovering >20 | 5 | 32.7 % (feasible+2pt) | border_band | 0.849 | không (0.0 % mẫu boot) | 0.096 | 0.520 | -0.423 | [—; -0.026] | − |
| hovering >20 | 5 | 32.7 % (feasible+2pt) | orb_lite_diff | 0.829 | không (0.0 % mẫu boot) | 0.205 | 0.520 | -0.314 | [—; +0.109] | 0 (CI chứa 0) |
| hovering >20 | 5 | 32.7 % (feasible+2pt) | raw_diff | 0.831 | không (0.0 % mẫu boot) | 0.182 | 0.520 | -0.338 | [—; +0.097] | 0 (CI chứa 0) |
| hovering >20 | 5 | 32.7 % (feasible+2pt) | tiny_det | 0.874 | không (0.0 % mẫu boot) | 0.213 | 0.520 | -0.306 | [—; +0.116] | 0 (CI chứa 0) |
| hovering >20 | 5 | 10.0 % (fixed) | border_band | 0.849 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.120] | − |
| hovering >20 | 5 | 10.0 % (fixed) | orb_lite_diff | 0.829 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; -0.006] | − |
| hovering >20 | 5 | 10.0 % (fixed) | raw_diff | 0.831 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; -0.015] | − |
| hovering >20 | 5 | 10.0 % (fixed) | tiny_det | 0.874 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; +0.025] | 0 (CI chứa 0) |
| hovering >20 | 5 | 15.0 % (fixed) | border_band | 0.849 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.066] | − |
| hovering >20 | 5 | 15.0 % (fixed) | orb_lite_diff | 0.829 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; +0.056] | 0 (CI chứa 0) |
| hovering >20 | 5 | 15.0 % (fixed) | raw_diff | 0.831 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; +0.044] | 0 (CI chứa 0) |
| hovering >20 | 5 | 15.0 % (fixed) | tiny_det | 0.874 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; +0.072] | 0 (CI chứa 0) |
| hovering >20 | 5 | 20.0 % (fixed) | border_band | 0.849 | không (0.0 % mẫu boot) | 0.096 | ∞ | −∞ | [—; -0.055] | − |
| hovering >20 | 5 | 20.0 % (fixed) | orb_lite_diff | 0.829 | không (0.0 % mẫu boot) | 0.205 | ∞ | −∞ | [—; +0.071] | 0 (CI chứa 0) |
| hovering >20 | 5 | 20.0 % (fixed) | raw_diff | 0.831 | không (0.0 % mẫu boot) | 0.182 | ∞ | −∞ | [—; +0.065] | 0 (CI chứa 0) |
| hovering >20 | 5 | 20.0 % (fixed) | tiny_det | 0.874 | không (0.0 % mẫu boot) | 0.213 | ∞ | −∞ | [—; +0.090] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 13.3 % (feasible+2pt) | border_band | 0.848 | không (0.0 % mẫu boot) | 0.113 | 0.563 | -0.450 | [—; -0.095] | − |
| moving 6-20 | 3 | 13.3 % (feasible+2pt) | orb_lite_diff | 0.705 | không (0.0 % mẫu boot) | 0.272 | 0.563 | -0.291 | [—; +0.110] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 13.3 % (feasible+2pt) | raw_diff | 0.605 | không (0.0 % mẫu boot) | 0.387 | 0.563 | -0.176 | [—; +0.132] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 13.3 % (feasible+2pt) | tiny_det | 0.751 | không (0.1 % mẫu boot) | 0.345 | 0.563 | -0.217 | [—; +0.347] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 10.0 % (fixed) | border_band | 0.848 | không (0.0 % mẫu boot) | 0.113 | ∞ | −∞ | [—; -0.105] | − |
| moving 6-20 | 3 | 10.0 % (fixed) | orb_lite_diff | 0.705 | không (0.0 % mẫu boot) | 0.272 | ∞ | −∞ | [—; +0.076] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 10.0 % (fixed) | raw_diff | 0.605 | không (0.0 % mẫu boot) | 0.387 | ∞ | −∞ | [—; +0.105] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 10.0 % (fixed) | tiny_det | 0.751 | không (0.0 % mẫu boot) | 0.345 | ∞ | −∞ | [—; +0.321] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 15.0 % (fixed) | border_band | 0.848 | không (0.0 % mẫu boot) | 0.113 | 0.452 | -0.340 | [—; -0.083] | − |
| moving 6-20 | 3 | 15.0 % (fixed) | orb_lite_diff | 0.705 | không (0.0 % mẫu boot) | 0.272 | 0.452 | -0.181 | [—; +0.118] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 15.0 % (fixed) | raw_diff | 0.605 | không (0.0 % mẫu boot) | 0.387 | 0.452 | -0.065 | [—; +0.155] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 15.0 % (fixed) | tiny_det | 0.751 | không (0.1 % mẫu boot) | 0.345 | 0.452 | -0.107 | [—; +0.352] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 20.0 % (fixed) | border_band | 0.848 | không (0.0 % mẫu boot) | 0.113 | 0.248 | -0.136 | [-0.520; -0.061] | − |
| moving 6-20 | 3 | 20.0 % (fixed) | orb_lite_diff | 0.705 | không (0.0 % mẫu boot) | 0.272 | 0.248 | +0.023 | [-0.333; +0.142] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 20.0 % (fixed) | raw_diff | 0.605 | không (0.0 % mẫu boot) | 0.387 | 0.248 | +0.139 | [-0.133; +0.277] | 0 (CI chứa 0) |
| moving 6-20 | 3 | 20.0 % (fixed) | tiny_det | 0.751 | không (0.3 % mẫu boot) | 0.345 | 0.248 | +0.097 | [-0.356; +0.376] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.6 % (feasible+2pt) | border_band | 0.796 | không (0.0 % mẫu boot) | 0.113 | 0.463 | -0.351 | [—; -0.019] | − |
| moving 6-20 | 5 | 10.6 % (feasible+2pt) | orb_lite_diff | 0.651 | không (0.0 % mẫu boot) | 0.272 | 0.463 | -0.192 | [—; +0.177] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.6 % (feasible+2pt) | raw_diff | 0.550 | không (0.0 % mẫu boot) | 0.387 | 0.463 | -0.076 | [—; +0.203] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.6 % (feasible+2pt) | tiny_det | 0.687 | không (0.1 % mẫu boot) | 0.345 | 0.463 | -0.118 | [—; +0.421] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.0 % (fixed) | border_band | 0.796 | không (0.0 % mẫu boot) | 0.113 | 0.494 | -0.382 | [—; -0.022] | − |
| moving 6-20 | 5 | 10.0 % (fixed) | orb_lite_diff | 0.651 | không (0.0 % mẫu boot) | 0.272 | 0.494 | -0.223 | [—; +0.171] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.0 % (fixed) | raw_diff | 0.550 | không (0.0 % mẫu boot) | 0.387 | 0.494 | -0.107 | [—; +0.196] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 10.0 % (fixed) | tiny_det | 0.687 | không (0.1 % mẫu boot) | 0.345 | 0.494 | -0.149 | [—; +0.414] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 15.0 % (fixed) | border_band | 0.796 | không (0.0 % mẫu boot) | 0.113 | 0.255 | -0.143 | [-0.376; -0.000] | − |
| moving 6-20 | 5 | 15.0 % (fixed) | orb_lite_diff | 0.651 | không (0.0 % mẫu boot) | 0.272 | 0.255 | +0.016 | [-0.203; +0.200] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 15.0 % (fixed) | raw_diff | 0.550 | không (0.0 % mẫu boot) | 0.387 | 0.255 | +0.132 | [-0.046; +0.295] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 15.0 % (fixed) | tiny_det | 0.687 | không (0.3 % mẫu boot) | 0.345 | 0.255 | +0.090 | [-0.269; +0.443] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 20.0 % (fixed) | border_band | 0.796 | không (0.0 % mẫu boot) | 0.113 | 0.164 | -0.052 | [-0.189; +0.016] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 20.0 % (fixed) | orb_lite_diff | 0.651 | không (0.0 % mẫu boot) | 0.272 | 0.164 | +0.107 | [-0.025; +0.218] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 20.0 % (fixed) | raw_diff | 0.550 | không (0.1 % mẫu boot) | 0.387 | 0.164 | +0.223 | [-0.030; +0.412] | 0 (CI chứa 0) |
| moving 6-20 | 5 | 20.0 % (fixed) | tiny_det | 0.687 | không (0.3 % mẫu boot) | 0.345 | 0.164 | +0.181 | [-0.076; +0.449] | 0 (CI chứa 0) |

- Tỉ lệ ĐÚNG / KẾT LUẬN ĐƯỢC của họ xác nhận: **62/67 = 92.5 %** (P3); bỏ 24 ô H2: **(62 − 24)/(67 − 24) = 38/43 = 88.4 %**.

## [POST HOC — không thuộc họ xác nhận] E2 — recall theo khung kể từ onset r(k)

Mô hình dùng r hằng: yolo26s_1024 = 0.944, yolo26n_640 = 0.876 (mọi box VISIBLE, TRAIN).

| split | mức | loại | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | yolo26n_640 | all | 0.579 | 0.579 | 0.587 | 0.595 | 0.614 | 0.618 | 0.599 | 0.601 | 0.611 | 0.611 | 0.612 |
| test | yolo26n_640 | border | 0.776 | 0.771 | 0.787 | 0.784 | 0.801 | 0.813 | 0.801 | 0.810 | 0.812 | 0.812 | 0.825 |
| test | yolo26n_640 | interior | 0.423 | 0.428 | 0.429 | 0.446 | 0.468 | 0.468 | 0.443 | 0.440 | 0.457 | 0.458 | 0.447 |
| test | yolo26s_1024 | all | 0.731 | 0.725 | 0.742 | 0.739 | 0.748 | 0.755 | 0.751 | 0.754 | 0.760 | 0.751 | 0.765 |
| test | yolo26s_1024 | border | 0.850 | 0.854 | 0.853 | 0.859 | 0.858 | 0.876 | 0.869 | 0.881 | 0.883 | 0.867 | 0.877 |
| test | yolo26s_1024 | interior | 0.637 | 0.623 | 0.654 | 0.645 | 0.662 | 0.662 | 0.660 | 0.657 | 0.665 | 0.662 | 0.678 |
| train | yolo26n_640 | all | 0.695 | 0.705 | 0.724 | 0.699 | 0.730 | 0.736 | 0.720 | 0.736 | 0.725 | 0.718 | 0.738 |
| train | yolo26n_640 | border | 0.763 | 0.782 | 0.796 | 0.778 | 0.797 | 0.799 | 0.791 | 0.810 | 0.805 | 0.811 | 0.829 |
| train | yolo26n_640 | interior | 0.572 | 0.567 | 0.591 | 0.556 | 0.610 | 0.624 | 0.594 | 0.606 | 0.586 | 0.558 | 0.581 |
| train | yolo26s_1024 | all | 0.858 | 0.848 | 0.850 | 0.851 | 0.861 | 0.872 | 0.872 | 0.872 | 0.874 | 0.878 | 0.870 |
| train | yolo26s_1024 | border | 0.904 | 0.889 | 0.902 | 0.908 | 0.913 | 0.910 | 0.905 | 0.922 | 0.922 | 0.921 | 0.914 |
| train | yolo26s_1024 | interior | 0.774 | 0.776 | 0.758 | 0.750 | 0.769 | 0.804 | 0.815 | 0.783 | 0.791 | 0.803 | 0.794 |

Kích thước box GT trung vị √(w·h) (px) theo k:

| split | loại | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | all | 21.9 | 21.8 | 21.6 | 21.6 | 21.4 | 21.2 | 21.2 | 21.4 | 21.2 | 21.1 | 21.4 |
| test | border | 36.0 | 36.7 | 37.6 | 38.9 | 39.2 | 40.2 | 39.9 | 39.9 | 40.0 | 41.2 | 41.2 |
| test | interior | 15.5 | 15.5 | 15.5 | 15.5 | 15.5 | 15.9 | 15.7 | 15.9 | 15.9 | 15.9 | 16.0 |
| train | all | 27.7 | 28.1 | 28.6 | 28.7 | 28.9 | 29.0 | 29.2 | 29.2 | 29.0 | 29.2 | 29.3 |
| train | border | 32.5 | 33.2 | 33.6 | 34.1 | 34.5 | 34.6 | 35.1 | 35.2 | 35.1 | 35.2 | 35.5 |
| train | interior | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 |

## [POST HOC — không thuộc họ xác nhận] E3 — tiêu chí 2 (mô tả): q_o* quan sát vs Thm P3-iii

**no frozen predicted CI → not a criterion.** Cue mật độ tổng hợp (P2B-A1) trên detector thật TEST; điểm dự đoán = Thm P3-iii (r = 1, q_b = 0, c = 0, ρ1 TRAIN = 0.0314).

| nhóm M | M trung vị | n sự kiện | L | ε | sàn periodic | a_P(ε) | q_o* quan sát | CI 95 % | q_o* dự đoán | trong CI |
|---|---|---|---|---|---|---|---|---|---|---|
| low | 9 | 225 | 3 | 2.0 % (frozen 2 %) | 7.6 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0273 | không |
| low | 9 | 225 | 3 | 9.6 % (floor+2pt) | 7.6 % | 0.409 | 0.0489 | [0.0000; 0.0882] | 0.0246 | có |
| low | 9 | 225 | 5 | 2.0 % (frozen 2 %) | 6.2 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0161 | không |
| low | 9 | 225 | 5 | 8.2 % (floor+2pt) | 6.2 % | 0.321 | 0.0254 | [0.0000; 0.0755] | 0.0148 | có |
| mid | 15 | 181 | 3 | 2.0 % (frozen 2 %) | 17.7 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0165 | không |
| mid | 15 | 181 | 3 | 19.7 % (floor+2pt) | 17.7 % | 0.648 | 0.0663 | [0.0000; 0.0762] | 0.0127 | có |
| mid | 15 | 181 | 5 | 2.0 % (frozen 2 %) | 13.8 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0097 | không |
| mid | 15 | 181 | 5 | 15.8 % (floor+2pt) | 13.8 % | 0.495 | 0.0316 | [0.0000; 0.0818] | 0.0079 | có |
| high | 36 | 331 | 3 | 2.0 % (frozen 2 %) | 32.0 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0069 | không |
| high | 36 | 331 | 3 | 34.0 % (floor+2pt) | 32.0 % | 0.586 | 0.0197 | [0.0000; 0.0403] | 0.0041 | có |
| high | 36 | 331 | 5 | 2.0 % (frozen 2 %) | 30.5 % | ∞ | 0.0000 | [0.0000; 0.0000] | 0.0041 | không |
| high | 36 | 331 | 5 | 32.5 % (floor+2pt) | 30.5 % | 0.520 | 0.0132 | [0.0000; 0.0414] | 0.0024 | có |

## [POST HOC — không thuộc họ xác nhận] E4 — hiệu chuẩn mô hình (Δ_pred TRAIN vs Δ̂ TEST, 48 ô H3/H5 + H8)

| tập | n | slope | intercept | MAE | lệch TB (quan sát − dự đoán) | Spearman |
|---|---|---|---|---|---|---|
| H3/H5 + H8 | 48 | 1.230 | -0.010 | 0.112 | -0.036 | 0.785 (p 4e-11) |
| H3/H5 (Δ) | 24 | 0.567 | -0.238 | 0.163 | -0.133 | 0.623 (p 0.0012) |
| H8 (ΔΔ) | 24 | 0.363 | +0.074 | 0.062 | +0.062 | 0.198 (p 0.35) |

| nhóm | dự đoán | dấu quan sát (điểm, ngưỡng δ_min) | n |
|---|---|---|---|
| H3-H5 | + | <=0 | 2 |
| H3-H5 | <=0 | <=0 | 22 |
| H8 | + | + | 19 |
| H8 | <=0 | + | 5 |

- H3/H5: Δ̂ < Δ_pred ở 19/24 ô; trung vị (Δ̂ − Δ_pred) = -0.160 (mô hình đánh giá thấp mức thua của gate). Ví dụ L5 border_band × hovering × 6–20: Δ_pred -0.031 vs Δ̂ -0.221 [-0.322; +0.080].

## [POST HOC — không thuộc họ xác nhận] E5 — trôi kênh TRAIN → TEST tại θ* đóng băng

| split | λ /khung | ρ1 | ρ3 | ρ5 | ρ10 | E3 (mọi onset) |
|---|---|---|---|---|---|---|
| train | 0.0363 | 0.0314 | 0.0896 | 0.1447 | 0.2666 | 865 |
| test | 0.0506 | 0.0382 | 0.1119 | 0.1822 | 0.3347 | 839 |

w = 5 (θ*):

| cue | ego | q_in TRAIN | q_in TEST | q_out TRAIN | q_out TEST | J TRAIN | J TEST | a_G dự đoán TRAIN | a_G dự đoán TEST |
|---|---|---|---|---|---|---|---|---|---|
| border_band | all | 0.1012 | 0.0543 | 0.0650 | 0.0431 | +0.0362 | +0.0111 | 0.1272 | 0.1029 |
| border_band | hovering | 0.0700 | 0.0490 | 0.0310 | 0.0359 | +0.0390 | +0.0131 | 0.0949 | 0.0965 |
| border_band | moving | 0.1885 | 0.0631 | 0.1484 | 0.0535 | +0.0401 | +0.0096 | 0.2080 | 0.1126 |
| orb_lite_diff | all | 0.1441 | 0.1154 | 0.0925 | 0.1020 | +0.0516 | +0.0135 | 0.2276 | 0.2319 |
| orb_lite_diff | hovering | 0.1058 | 0.0912 | 0.0609 | 0.0736 | +0.0449 | +0.0176 | 0.1963 | 0.2054 |
| orb_lite_diff | moving | 0.2514 | 0.1563 | 0.1703 | 0.1430 | +0.0811 | +0.0133 | 0.3061 | 0.2714 |
| raw_diff | all | 0.2493 | 0.2319 | 0.1917 | 0.2119 | +0.0576 | +0.0200 | 0.2492 | 0.2642 |
| raw_diff | hovering | 0.1652 | 0.1318 | 0.1290 | 0.1296 | +0.0362 | +0.0022 | 0.1858 | 0.1815 |
| raw_diff | moving | 0.4851 | 0.4005 | 0.3457 | 0.3307 | +0.1394 | +0.0698 | 0.4080 | 0.3869 |
| tiny_det | all | 0.1113 | 0.1174 | 0.0630 | 0.1097 | +0.0483 | +0.0077 | 0.2268 | 0.2665 |
| tiny_det | hovering | 0.1251 | 0.0733 | 0.0765 | 0.0520 | +0.0486 | +0.0213 | 0.2400 | 0.2133 |
| tiny_det | moving | 0.0728 | 0.1918 | 0.0299 | 0.1931 | +0.0429 | -0.0013 | 0.1935 | 0.3455 |

q_in theo tầng (cửa sổ w = 5 của sự kiện E3 cửa sổ sinh thuộc tầng), q_out của ego, P(cue bắn ≥ 1 lần ở k = 0…3):

| cue | tầng | split | n sự kiện | q_in tầng | q_out ego | J tầng | P(bắn, k ≤ 3) |
|---|---|---|---|---|---|---|---|
| border_band | hovering 6-20 | test | 121 | 0.1113 | 0.0359 | +0.0754 | 0.149 |
| border_band | hovering 6-20 | train | 215 | 0.0413 | 0.0310 | +0.0103 | 0.060 |
| border_band | hovering >20 | test | 332 | 0.0293 | 0.0359 | -0.0066 | 0.039 |
| border_band | hovering >20 | train | 310 | 0.1087 | 0.0310 | +0.0777 | 0.135 |
| border_band | moving 6-20 | test | 221 | 0.0418 | 0.0535 | -0.0117 | 0.054 |
| border_band | moving 6-20 | train | 167 | 0.1502 | 0.1484 | +0.0018 | 0.222 |
| orb_lite_diff | hovering 6-20 | test | 121 | 0.1792 | 0.0736 | +0.1057 | 0.264 |
| orb_lite_diff | hovering 6-20 | train | 215 | 0.0646 | 0.0609 | +0.0037 | 0.126 |
| orb_lite_diff | hovering >20 | test | 332 | 0.0449 | 0.0736 | -0.0287 | 0.084 |
| orb_lite_diff | hovering >20 | train | 310 | 0.1248 | 0.0609 | +0.0639 | 0.139 |
| orb_lite_diff | moving 6-20 | test | 221 | 0.1861 | 0.1430 | +0.0431 | 0.240 |
| orb_lite_diff | moving 6-20 | train | 167 | 0.2307 | 0.1703 | +0.0603 | 0.359 |
| raw_diff | hovering 6-20 | test | 121 | 0.3132 | 0.1296 | +0.1836 | 0.388 |
| raw_diff | hovering 6-20 | train | 215 | 0.1155 | 0.1290 | -0.0136 | 0.140 |
| raw_diff | hovering >20 | test | 332 | 0.0568 | 0.1296 | -0.0728 | 0.081 |
| raw_diff | hovering >20 | train | 310 | 0.1766 | 0.1290 | +0.0475 | 0.200 |
| raw_diff | moving 6-20 | test | 221 | 0.3747 | 0.3307 | +0.0440 | 0.380 |
| raw_diff | moving 6-20 | train | 167 | 0.4507 | 0.3457 | +0.1050 | 0.491 |
| tiny_det | hovering 6-20 | test | 121 | 0.2264 | 0.0520 | +0.1744 | 0.397 |
| tiny_det | hovering 6-20 | train | 215 | 0.0869 | 0.0765 | +0.0104 | 0.121 |
| tiny_det | hovering >20 | test | 332 | 0.0027 | 0.0520 | -0.0493 | 0.006 |
| tiny_det | hovering >20 | train | 310 | 0.1537 | 0.0765 | +0.0772 | 0.255 |
| tiny_det | moving 6-20 | test | 221 | 0.1734 | 0.1931 | -0.0197 | 0.158 |
| tiny_det | moving 6-20 | train | 167 | 0.0789 | 0.0299 | +0.0491 | 0.156 |

Ô border_band (matched cost, R = 30): dự đoán TRAIN vs quan sát TEST:

| tầng | L | miss_P dự đoán | miss_P TEST | miss_G dự đoán | miss_G TEST | Δ_pred | Δ̂ [CI] | a_G dự đoán | a_G TEST |
|---|---|---|---|---|---|---|---|---|---|
| hovering 6-20 | 3 | 0.635 | 0.649 | 0.594 | 0.748 | +0.040 | -0.099 [-0.210; +0.220] | 0.099 | 0.096 |
| hovering 6-20 | 5 | 0.456 | 0.476 | 0.486 | 0.697 | -0.031 | -0.221 [-0.322; +0.080] | 0.099 | 0.096 |
| hovering >20 | 3 | 0.630 | 0.759 | 0.590 | 0.893 | +0.040 | -0.134 [-0.193; -0.098] | 0.099 | 0.096 |
| hovering >20 | 5 | 0.452 | 0.639 | 0.482 | 0.849 | -0.030 | -0.209 [-0.296; -0.158] | 0.099 | 0.096 |
| moving 6-20 | 3 | 0.266 | 0.638 | 0.589 | 0.848 | -0.323 | -0.210 [-0.277; -0.128] | 0.196 | 0.113 |
| moving 6-20 | 5 | 0.056 | 0.454 | 0.432 | 0.796 | -0.376 | -0.342 [-0.421; -0.247] | 0.196 | 0.113 |

## [POST HOC — không thuộc họ xác nhận] E6 — bảng cho bài (lấy từ P3, không tính lại)

(a) Họ xác nhận, 3 mức theo nhóm:

| nhóm | n | ĐÚNG | SAI | CHƯA KẾT LUẬN | ĐÚNG / KẾT LUẬN ĐƯỢC |
|---|---|---|---|---|---|
| H2 | 24 | 24 | 0 | 0 | 100.0 % |
| H3-H5 | 24 | 21 | 1 | 2 | 95.5 % |
| H8 | 24 | 17 | 4 | 3 | 81.0 % |
| tổng | 72 | 62 | 5 | 5 | 92.5 % |
| tổng, loại H2 | 48 | 38 | 5 | 5 | 88.4 % |

(b) 21 ô dự đoán "+":

| nhóm | KPI | ego | M_bin | cue | giá trị dự đoán | quan sát | CI 95 % | kết quả |
|---|---|---|---|---|---|---|---|---|
| H3-H5 | 3 | hovering | 6-20 | border_band | +0.0401 | -0.0988 | [-0.2100; +0.2198] | CHƯA KẾT LUẬN |
| H3-H5 | 3 | hovering | >20 | border_band | +0.0402 | -0.1342 | [-0.1935; -0.0978] | SAI |
| H8 | H8_L5 | hovering | 6-20 | border_band | +0.0112 | +0.0430 | [+0.0087; +0.0617] | ĐÚNG |
| H8 | H8_L5 | hovering | >20 | border_band | +0.0112 | +0.0494 | [+0.0200; +0.1072] | ĐÚNG |
| H8 | H8_L3 | moving | 6-20 | border_band | +0.0281 | +0.0473 | [+0.0062; +0.0824] | ĐÚNG |
| H8 | H8_L5 | moving | 6-20 | border_band | +0.0282 | +0.0778 | [+0.0090; +0.1252] | ĐÚNG |
| H8 | H8_L3 | hovering | 6-20 | orb_lite_diff | +0.0254 | +0.0731 | [+0.0111; +0.1254] | ĐÚNG |
| H8 | H8_L5 | hovering | 6-20 | orb_lite_diff | +0.0251 | +0.0781 | [+0.0032; +0.1434] | ĐÚNG |
| H8 | H8_L3 | hovering | >20 | orb_lite_diff | +0.0258 | +0.0979 | [+0.0386; +0.2278] | ĐÚNG |
| H8 | H8_L5 | hovering | >20 | orb_lite_diff | +0.0256 | +0.1068 | [+0.0315; +0.2537] | ĐÚNG |
| H8 | H8_L3 | moving | 6-20 | orb_lite_diff | +0.0298 | +0.1299 | [+0.0201; +0.2227] | ĐÚNG |
| H8 | H8_L3 | hovering | 6-20 | raw_diff | +0.0210 | +0.0459 | [+0.0091; +0.0688] | ĐÚNG |
| H8 | H8_L5 | hovering | 6-20 | raw_diff | +0.0285 | +0.0727 | [+0.0114; +0.1039] | ĐÚNG |
| H8 | H8_L3 | hovering | >20 | raw_diff | +0.0214 | +0.0824 | [+0.0367; +0.1859] | ĐÚNG |
| H8 | H8_L5 | hovering | >20 | raw_diff | +0.0290 | +0.1060 | [+0.0362; +0.2440] | ĐÚNG |
| H8 | H8_L3 | moving | 6-20 | raw_diff | +0.0080 | +0.1070 | [+0.0085; +0.2142] | ĐÚNG |
| H8 | H8_L3 | hovering | 6-20 | tiny_det | +0.0365 | +0.0512 | [-0.0139; +0.0997] | CHƯA KẾT LUẬN |
| H8 | H8_L5 | hovering | 6-20 | tiny_det | +0.0162 | +0.0533 | [-0.0215; +0.0957] | CHƯA KẾT LUẬN |
| H8 | H8_L3 | hovering | >20 | tiny_det | +0.0342 | +0.1065 | [+0.0473; +0.2413] | ĐÚNG |
| H8 | H8_L3 | moving | 6-20 | tiny_det | +0.0326 | +0.1285 | [+0.0248; +0.2096] | ĐÚNG |
| H8 | H8_L5 | moving | 6-20 | tiny_det | +0.0218 | +0.1036 | [+0.0112; +0.1796] | ĐÚNG |

(c) Biến thể mô tả trên 72 ô (P3 describe_test.json; H2 chỉ theo dấu):

| biến thể | ĐÚNG | SAI | CHƯA KẾT LUẬN |
|---|---|---|---|
| onset_alt | 43 | 3 | 26 |
| roi5 | 63 | 5 | 4 |
| score005 | 48 | 1 | 23 |
| score050 | 63 | 6 | 3 |

Hình: paper/figs/fig_p3b_calibration.pdf, fig_p3b_recall_vs_k.pdf, fig_p3b_iso_feasible.pdf.
