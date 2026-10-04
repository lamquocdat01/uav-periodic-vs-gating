# PREREG_28 — sinh tự động (CHƯA đóng băng)

*Sinh bởi `code/p2/p2_prereg_build.py` — 2026-10-01 16:54. Kênh: split=train, simulated=False. KHÔNG hash/tag: đóng băng (sha256 + git tag) là quyết định của anh Đạt, sau khi kênh đo thật trên TRAIN.*

Công thức: THEORY_28 §5–§6, chế độ khung nền = meanfield. Luật họ + θ* + tiêu chí bác bỏ: PREREG_28_DRAFT.md §2, §4 (quyết định 01-10-2026: PREREG_28_DRAFT.md §Quyết định).

## §0 Trạng thái kênh: v2 — r ở score ≥ 0.25; E3: không lọc — E3 đầy đủ theo định nghĩa P0 (P2G-Q5; audit khung trước onset chỉ mô tả, PREREG §5); 865 sự kiện E3 TRAIN (VISIBLE chính, trước cắt cửa sổ sinh). Bản kênh v1 (score 0,05): results/p2/channel_train_score005.json (phụ lục).

## §1 Dữ liệu (luật cố định, áp cho cả TEST khi mở)

- M0207: 885 ảnh nhưng GT chỉ tới khung 571 → chỉ dùng khung ≤ 571 (khung GT cuối); khung ngoài GT không có nhãn, không phải "0 xe".
- M0901: ảnh thật 960 × 540 (GT x+w tối đa 959) → dải biên (border) tính theo bề rộng 960; mọi chuỗi khác dùng kích thước đọc từ ảnh.
- E3 = định nghĩa P0 đầy đủ (VISIBLE: occlusion ≠ 2 và out_of_view ≠ 2), **không loại sự kiện nào**; audit khung trước onset chỉ mô tả (§5).

## §2 Luật cố định (trước freeze)

- **Ngưỡng score "phát hiện"**: chính = 0.25 (mặc định Ultralytics predict, điểm vận hành); 0.05, 0.5 = phụ lục mô tả. Cue tiny_det giữ θ* riêng (ngưỡng trên score của cue, không phải ngưỡng detector).
- **Luật chi phí**: cue có c ≥ 1 bị loại trước khi chọn θ* (Thm G3: c ≥ 1 ⇒ gate không thể rẻ hơn tuần hoàn). Đã loại: ego_comp_diff (c = 1.02 ≥ 1 (luật chi phí, Thm G3)).
- **Cột "dự đoán"** (matched cost, H3/H5): ghi "+" iff Δ_pred = miss_P − miss_G > δ_min, **δ_min = 0.005** (tuyệt đối, đơn vị tỉ lệ miss); ngược lại "≤0". H8: "+" iff ΔΔ_pred = Δ_pred(r thấp) − Δ_pred(r chính) > δ_min. Iso-KPI (H2): "+" iff miss_G ≤ ε và a_G < a_P(ε).
  - Δ_pred (H3/H5, họ xác nhận): lớn nhất +0.0402, nhỏ nhất -0.5065; số ô "+": 2/24.
  - ΔΔ_pred (H8, mọi ô R = 30 kể cả mô tả): lớn nhất +0.0384, nhỏ nhất -0.0222; số ô "+": 200/468.
- **H8 (Q2)**: mức recall thấp hợp lệ khi r_main − r_low ≥ 0.05 (score 0.25, TRAIN, ước lượng điểm, IoU 0.5). Ứng viên: yolo26n_1024, yolo26s_640, yolo26n_640. Cặp H8 = (yolo26s_1024, mức hợp lệ có r thấp nhất), tối đa 2 cặp. Không mức nào hợp lệ → H8 ra khỏi họ xác nhận (mô tả). Không train thêm detector.

| mức | r (score chính) | CI 95 % | r_main − r | hợp lệ |
|---|---|---|---|---|
| yolo26s_1024 | 0.9440 | [0.9130; 0.9647] | — | chính |
| yolo26n_1024 | 0.9212 | [0.8876; 0.9464] | 0.0228 | không |
| yolo26s_640 | 0.9309 | [0.8969; 0.9559] | 0.0131 | không |
| yolo26n_640 | 0.8762 | [0.8290; 0.9122] | 0.0678 | có |

**Kết luận H8:** cặp yolo26s_1024>yolo26n_640 vào họ xác nhận.

- **r† cho H8 (P2H-Q7)**: r† = argmax_r Δ_pred(r) trên [0, 1] cho từng cấu hình (cue, θ*, tầng ego × M, L), cùng mô hình dự đoán của ô (Remark G2′: Δ lõm, Δ(0) = 0 ⇒ Δ không tăng trên [r†, 1]). Ô H8 chỉ giữ khi r_main > r† và r_low > r†; ngược lại dự đoán "—" và ra khỏi họ xác nhận. r† = 0 nghĩa là Δ_pred(r) không tăng trên toàn [0, 1].
  - r_main = 0.944, r_low = 0.876; r† lớn nhất = 0.723; số ô H8 vi phạm: 0/36 (ô H8 đủ cỡ mẫu, kể cả L = 10 đã bị thu gọn).

| cue | ego | M_bin | L | r† | r_main > r† và r_low > r† |
|---|---|---|---|---|---|
| border_band | hovering | 6-20 | 3 | 0.723 | có |
| border_band | hovering | 6-20 | 5 | 0.357 | có |
| border_band | hovering | 6-20 | 10 | 0.013 | có |
| border_band | hovering | >20 | 3 | 0.719 | có |
| border_band | hovering | >20 | 5 | 0.360 | có |
| border_band | hovering | >20 | 10 | 0.016 | có |
| border_band | moving | 6-20 | 3 | 0.000 | có |
| border_band | moving | 6-20 | 5 | 0.000 | có |
| border_band | moving | 6-20 | 10 | 0.000 | có |
| orb_lite_diff | hovering | 6-20 | 3 | 0.000 | có |
| orb_lite_diff | hovering | 6-20 | 5 | 0.000 | có |
| orb_lite_diff | hovering | 6-20 | 10 | 0.000 | có |
| orb_lite_diff | hovering | >20 | 3 | 0.000 | có |
| orb_lite_diff | hovering | >20 | 5 | 0.000 | có |
| orb_lite_diff | hovering | >20 | 10 | 0.000 | có |
| orb_lite_diff | moving | 6-20 | 3 | 0.000 | có |
| orb_lite_diff | moving | 6-20 | 5 | 0.000 | có |
| orb_lite_diff | moving | 6-20 | 10 | 0.000 | có |
| raw_diff | hovering | 6-20 | 3 | 0.386 | có |
| raw_diff | hovering | 6-20 | 5 | 0.272 | có |
| raw_diff | hovering | 6-20 | 10 | 0.129 | có |
| raw_diff | hovering | >20 | 3 | 0.384 | có |
| raw_diff | hovering | >20 | 5 | 0.272 | có |
| raw_diff | hovering | >20 | 10 | 0.131 | có |
| raw_diff | moving | 6-20 | 3 | 0.000 | có |
| raw_diff | moving | 6-20 | 5 | 0.000 | có |
| raw_diff | moving | 6-20 | 10 | 0.000 | có |
| tiny_det | hovering | 6-20 | 3 | 0.000 | có |
| tiny_det | hovering | 6-20 | 5 | 0.000 | có |
| tiny_det | hovering | 6-20 | 10 | 0.000 | có |
| tiny_det | hovering | >20 | 3 | 0.000 | có |
| tiny_det | hovering | >20 | 5 | 0.000 | có |
| tiny_det | hovering | >20 | 10 | 0.000 | có |
| tiny_det | moving | 6-20 | 3 | 0.000 | có |
| tiny_det | moving | 6-20 | 5 | 0.000 | có |
| tiny_det | moving | 6-20 | 10 | 0.000 | có |

- **Luật chấm 3 mức (P2H-Q6, thay DRAFT §4 "Scoring")** — CI 95 % = bootstrap ghép cặp theo chuỗi, B = 1000, seed 42:
  - dự đoán "+": ĐÚNG iff CI_dưới(Δ̂) > 0; SAI iff CI_trên ≤ 0; còn lại CHƯA KẾT LUẬN.
  - dự đoán "≤0": ĐÚNG iff CI_trên < δ_min (0.005); SAI iff CI_dưới > δ_min; còn lại CHƯA KẾT LUẬN.
  - H8: cùng luật áp lên ΔΔ̂. Iso-KPI (H2): ĐÚNG/SAI theo "gate rẻ hơn ở KPI bằng nhau"; CHƯA KẾT LUẬN khi CI của (a_G − a_P(ε)) chứa 0.
  - Tiêu chí bác bỏ (i): < 80 % ô KẾT LUẬN ĐƯỢC là đúng; (ii) q_o* ngoài CI dự đoán ở ≥ 2/3 nhóm M; (iii) H8 sai ở ≥ 50 % ô H8 kết luận được.
  - **Tiêu chí công suất P**: > 50 % ô họ xác nhận CHƯA KẾT LUẬN → kết luận bài = "chưa kiểm được" (không xác nhận, không bác bỏ); báo tỉ lệ 3 mức theo H2 / H3–H5 / H8. Holm chỉ để báo cáo. Code: `code/p2/p2_prereg_score.py`.

## §3 Ngưỡng hovering (ego-motion)

- Biến: dịch tâm ảnh qua homography khung liền kề (ORB + RANSAC), **median theo chuỗi**, px/khung; mô hình trên **log10** của biến. Chỉ chuỗi TRAIN.
- Phương pháp chính: GMM 2 thành phần (EM, seed 42) trên log10 → ngưỡng = điểm hai thành phần (có trọng số) bằng nhau: **0.61 px/khung**; mode 0.129 / 1.203 px, Ashman D = 2.92 (> 2 ⇒ bimodal) → **19 hovering / 11 moving** chuỗi.
- Đã thử: khe lớn nhất giữa hai giá trị liên tiếp (mỗi phía ≥ 3 chuỗi): khe 0.24 log10 (< 0,3) tại 0.090 px (8/22 chuỗi) → **không bimodal theo khe, không dùng**.
- Độ nhạy ngưỡng ±25 % = phụ lục mô tả (không vào họ xác nhận):
  - ngưỡng × 0.75 = 0.46 px → 18 hovering / 12 moving.
  - ngưỡng × 1 = 0.61 px → 19 hovering / 11 moving.
  - ngưỡng × 1.25 = 0.76 px → 20 hovering / 10 moving.

## §4 Chi phí cue c

- c = t_cue / t_det, **tử số và mẫu số cùng một phiên đo** (bench_c_real.json: 200 cặp khung TRAIN thật, 30 chuỗi, 3 lần, median): mẫu số yolo26s@1024 iGPU (Ultralytics predict) = **94.6 ms** (các lần: 94.6, 97.5, 80.1 ms). Cue OpenCV đo **1 luồng**; tiny_det = yolo26n@320 CPU.
- Dải mẫu số yolo26s@1024 iGPU qua 3 phiên: **62.9–94.6 ms** (2026-09-29 bench_openvino (P2D-C1): 62.9; 2026-09-30 bench_openvino (P2E-B1a): 77.5; 2026-09-30 bench_c_real (P2E-B1b, mẫu số c): 94.6). Chỉ báo dải; c dùng phiên cùng tử số.
- c theo cue (median): raw_diff 0.023; ego_comp_diff 1.024; orb_lite_diff 0.098; border_band 0.026; tiny_det_yolo26n_320 0.126.

## §5 Audit khung trước onset — CHỈ MÔ TẢ, KHÔNG loại sự kiện (luật cố định, áp cho TEST sau freeze)

- Mỗi sự kiện E3: b = khung onset (khung VISIBLE đầu tiên); box GT của track ở khung b; dump yolo26s_1024 (score ≥ 0.25) ở 5 khung b−5…b−1; khớp = IoU ≥ 0.3 với box GT khung b. b−5 < khung đầu → censored-start; ≤ 2/5 khớp → true-birth. Với ≥ 3/5 khớp: **partial-entry** = border và (box khung b cách mép ảnh thật ≤ 2 px hoặc track có GT ở b−5…b−1 với out_of_view = 2); **visibility-transition** = track có GT ở các khung đó với occlusion = 2 (hoặc out_of_view = 2 khi interior); **annotation-late** (gán trễ THẬT) = không có dòng GT nào của track trong các khung đó; còn lại → true-birth.
- **Không loại sự kiện nào** (P2G-Q5). Nếu annotation-late THẬT > 20 % interior trên TRAIN → ghi hạn chế ở §6, vẫn không loại.

| TRAIN, E3 VISIBLE chính | partial-entry | visibility-transition | annotation-late | true-birth | censored-start | n | % annotation-late |
|---|---|---|---|---|---|---|---|
| border | 466 | 4 | 30 | 72 | 2 | 574 | 5.2 % |
| interior | 0 | 52 | 197 | 42 | 0 | 291 | 67.7 % |
| all | 466 | 56 | 227 | 114 | 2 | 865 | 26.2 % |

Độ nhạy ngưỡng mép (P2H-Q9, chỉ mô tả): partial-entry khi cách mép ≤ 5 px thay vì ≤ 2 px:

| TRAIN, mép ≤ 5 px | partial-entry | visibility-transition | annotation-late | true-birth | censored-start | n | % annotation-late |
|---|---|---|---|---|---|---|---|
| border | 484 | 4 | 12 | 72 | 2 | 574 | 2.1 % |
| interior | 0 | 52 | 197 | 42 | 0 | 291 | 67.7 % |
| all | 484 | 56 | 209 | 114 | 2 | 865 | 24.2 % |

- Placebo (box cùng cỡ đặt ngẫu nhiên ở khung b−3, seed 42): khớp 3.1 % (863 sự kiện) ⇒ các khớp là vật thể thật, không do mật độ box.
- Kiểm phụ (model-assisted (Claude xem crop start−2 … start+1), không phải nhãn người): đúng nhãn 5 lớp 16/23 = 69.6 %; nhị phân "đã thấy xe trước onset" 19/23 = 82.6 %. Bất đồng: A04 (border, tự động true-birth 2/5, mắt partial-entry), A07 (border, tự động true-birth 0/5, mắt partial-entry), A08 (border, tự động true-birth 2/5, mắt partial-entry), A09 (border, tự động annotation-late 4/5, mắt partial-entry), A10 (border, tự động true-birth 2/5, mắt partial-entry), A17 (border, tự động annotation-late 5/5, mắt partial-entry), A18 (border, tự động partial-entry 5/5, mắt visibility-transition) — không đổi nhãn tự động. Chi tiết: AUDIT_SHORT_E3.md.

## §6 Điều đã biết trên TRAIN trước freeze (quan sát TRAIN — TEST CHƯA MỞ)

- Kênh khởi phát yếu: J = q_in − q_out tại θ* ≤ 0.058 ở mọi cue (border_band 0.036, orb_lite_diff 0.052, raw_diff 0.058, tiny_det 0.048).
- Ràng buộc a_G ≤ 0.25 chặn θ* (θ* ≠ argmax J không ràng buộc) ở: border_band (a_G = 0.127), orb_lite_diff (a_G = 0.228), raw_diff (a_G = 0.249), tiny_det (a_G = 0.227).
- Dự đoán gate thắng ("+") chỉ ở 21/72 ô họ xác nhận: H3/H5 border_band × hovering × M 6-20 × 3; H3/H5 border_band × hovering × M >20 × 3; H8 border_band × hovering × M 6-20 × H8_L5; H8 border_band × hovering × M >20 × H8_L5; H8 border_band × moving × M 6-20 × H8_L3; H8 border_band × moving × M 6-20 × H8_L5; H8 orb_lite_diff × hovering × M 6-20 × H8_L3; H8 orb_lite_diff × hovering × M 6-20 × H8_L5; H8 orb_lite_diff × hovering × M >20 × H8_L3; H8 orb_lite_diff × hovering × M >20 × H8_L5; H8 orb_lite_diff × moving × M 6-20 × H8_L3; H8 raw_diff × hovering × M 6-20 × H8_L3; H8 raw_diff × hovering × M 6-20 × H8_L5; H8 raw_diff × hovering × M >20 × H8_L3; H8 raw_diff × hovering × M >20 × H8_L5; H8 raw_diff × moving × M 6-20 × H8_L3; H8 tiny_det × hovering × M 6-20 × H8_L3; H8 tiny_det × hovering × M 6-20 × H8_L5; H8 tiny_det × hovering × M >20 × H8_L3; H8 tiny_det × moving × M 6-20 × H8_L3; H8 tiny_det × moving × M 6-20 × H8_L5.
- **Hạn chế dữ liệu (không loại sự kiện)**: 67.7 % sự kiện E3 interior (TRAIN) là annotation-late THẬT (> 20 %): detector đã thấy xe ≥ 3/5 khung trước onset mà track chưa có dòng GT nào; border: 5.2 %. Onset GT của các sự kiện này có thể muộn hơn lúc xe thật sự xuất hiện → xem phụ lục onset_alt.
- **Bản 01-10-2026 sáng (P2F: lọc annotation-late, loại 82 % E3 TRAIN) bị RÚT vì luật audit sai** (gộp xe vào từ mép lộ một phần và track đã có GT với occlusion/out_of_view = 2 — chính định nghĩa onset của P0 — thành "gán trễ"); **không số nào từ bản đó được dùng**.
- Mọi điều trên là quan sát trên TRAIN dùng để dựng dự đoán; TEST chưa mở.

## Phụ lục — độ nhạy onset (mô tả, không vào họ xác nhận)

onset_alt = khung sớm nhất trong b−5…b−1 có detector khớp (không có → b). TRAIN: 821 sự kiện dời sớm, trung bình 4.33 khung. Dự đoán matched cost (H3/H5) ở ô xác nhận, onset GT vs onset_alt (cửa sổ KPI T = min(D + shift, L + 1)):

| cue | ego | M_bin | L | Δ_pred onset GT | Δ_pred onset_alt | thay đổi | dự đoán GT → alt |
|---|---|---|---|---|---|---|---|
| border_band | hovering | 6-20 | 3 | +0.0401 | +0.0402 | +0.0001 | + → + |
| border_band | hovering | 6-20 | 5 | -0.0306 | -0.0333 | -0.0027 | <=0 → <=0 |
| border_band | hovering | >20 | 3 | +0.0402 | +0.0403 | +0.0001 | + → + |
| border_band | hovering | >20 | 5 | -0.0301 | -0.0338 | -0.0037 | <=0 → <=0 |
| border_band | moving | 6-20 | 3 | -0.3231 | -0.3246 | -0.0014 | <=0 → <=0 |
| border_band | moving | 6-20 | 5 | -0.3760 | -0.3771 | -0.0011 | <=0 → <=0 |
| orb_lite_diff | hovering | 6-20 | 3 | -0.2413 | -0.2475 | -0.0061 | <=0 → <=0 |
| orb_lite_diff | hovering | 6-20 | 5 | -0.3094 | -0.3182 | -0.0088 | <=0 → <=0 |
| orb_lite_diff | hovering | >20 | 3 | -0.2448 | -0.2485 | -0.0037 | <=0 → <=0 |
| orb_lite_diff | hovering | >20 | 5 | -0.3131 | -0.3194 | -0.0063 | <=0 → <=0 |
| orb_lite_diff | moving | 6-20 | 3 | -0.4366 | -0.4393 | -0.0026 | <=0 → <=0 |
| orb_lite_diff | moving | 6-20 | 5 | -0.3116 | -0.3138 | -0.0023 | <=0 → <=0 |
| raw_diff | hovering | 6-20 | 3 | -0.0407 | -0.0444 | -0.0037 | <=0 → <=0 |
| raw_diff | hovering | 6-20 | 5 | -0.1291 | -0.1362 | -0.0071 | <=0 → <=0 |
| raw_diff | hovering | >20 | 3 | -0.0428 | -0.0450 | -0.0022 | <=0 → <=0 |
| raw_diff | hovering | >20 | 5 | -0.1313 | -0.1372 | -0.0058 | <=0 → <=0 |
| raw_diff | moving | 6-20 | 3 | -0.2682 | -0.2708 | -0.0025 | <=0 → <=0 |
| raw_diff | moving | 6-20 | 5 | -0.1438 | -0.1459 | -0.0021 | <=0 → <=0 |
| tiny_det | hovering | 6-20 | 3 | -0.4310 | -0.4405 | -0.0096 | <=0 → <=0 |
| tiny_det | hovering | 6-20 | 5 | -0.3885 | -0.3964 | -0.0079 | <=0 → <=0 |
| tiny_det | hovering | >20 | 3 | -0.5065 | -0.5123 | -0.0058 | <=0 → <=0 |
| tiny_det | hovering | >20 | 5 | -0.3908 | -0.3907 | +0.0001 | <=0 → <=0 |
| tiny_det | moving | 6-20 | 3 | -0.3752 | -0.3769 | -0.0017 | <=0 → <=0 |
| tiny_det | moving | 6-20 | 5 | -0.3962 | -0.3977 | -0.0015 | <=0 → <=0 |

## θ* (B1: argmax q_in − q_out ở w = 5, a_G(R = 30) ≤ 0.25; hoà → a_G nhỏ hơn)

| cue | số θ ứng viên | θ* | q_in | q_out | J | a_G | c | J không ràng buộc | ràng buộc a_G chặn | loại |
|---|---|---|---|---|---|---|---|---|---|---|
| border_band | 4 | 0.0768 | 0.1012 | 0.0650 | 0.0362 | 0.1272 | 0.026 | 0.0782 | có | — |
| ego_comp_diff | 3 | — | — | — | — | — | 1.024 | — | — | c = 1.02 ≥ 1 (luật chi phí, Thm G3) |
| orb_lite_diff | 4 | 0.0436 | 0.1441 | 0.0925 | 0.0516 | 0.2276 | 0.098 | 0.1340 | có | — |
| raw_diff | 4 | 0.0279 | 0.2493 | 0.1917 | 0.0576 | 0.2492 | 0.023 | 0.1182 | có | — |
| tiny_det | 4 | 0.8833 | 0.1113 | 0.0630 | 0.0483 | 0.2268 | 0.126 | 0.0767 | có | — |

## Họ xác nhận: 72 ô (giới hạn 80)

Detector chính: yolo26s_1024; cặp H8: yolo26s_1024>yolo26n_640; R = 30; ego: hovering, moving (sự kiện tách theo ego: có).

| bước | mô tả | số ô sau bước |
|---|---|---|
| start | — | 144 |
| drop_eps5 | bỏ ε = 5 % (giữ 2 %) | 108 |
| drop_L10 | bỏ L_max = 10 | 72 |
| H8_rdagger | bỏ ô H8 có r_main hoặc r_low ≤ r† (Q7) | 72 |

- Tổng ô (gồm mô tả): 4164; ô xác nhận: 72; dự đoán gate thắng (+) trong họ: 21.
- Theo giả thuyết (ô xác nhận): H2: 24, H3/H5: 24, H8: 24

| giả thuyết | KPI | ε | M_bin | ego | detector | cue | dự đoán | Δ dự đoán | n sự kiện / chuỗi (TRAIN) |
|---|---|---|---|---|---|---|---|---|---|
| H2 | iso_L3 | 0.02 | 6-20 | hovering | yolo26s_1024 | border_band | <=0 | — | 215 / 15 |
| H2 | iso_L5 | 0.02 | 6-20 | hovering | yolo26s_1024 | border_band | <=0 | — | 215 / 15 |
| H2 | iso_L3 | 0.02 | >20 | hovering | yolo26s_1024 | border_band | <=0 | — | 310 / 12 |
| H2 | iso_L5 | 0.02 | >20 | hovering | yolo26s_1024 | border_band | <=0 | — | 310 / 12 |
| H2 | iso_L3 | 0.02 | 6-20 | moving | yolo26s_1024 | border_band | <=0 | — | 167 / 10 |
| H2 | iso_L5 | 0.02 | 6-20 | moving | yolo26s_1024 | border_band | <=0 | — | 167 / 10 |
| H2 | iso_L3 | 0.02 | 6-20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | — | 215 / 15 |
| H2 | iso_L5 | 0.02 | 6-20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | — | 215 / 15 |
| H2 | iso_L3 | 0.02 | >20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | — | 310 / 12 |
| H2 | iso_L5 | 0.02 | >20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | — | 310 / 12 |
| H2 | iso_L3 | 0.02 | 6-20 | moving | yolo26s_1024 | orb_lite_diff | <=0 | — | 167 / 10 |
| H2 | iso_L5 | 0.02 | 6-20 | moving | yolo26s_1024 | orb_lite_diff | <=0 | — | 167 / 10 |
| H2 | iso_L3 | 0.02 | 6-20 | hovering | yolo26s_1024 | raw_diff | <=0 | — | 215 / 15 |
| H2 | iso_L5 | 0.02 | 6-20 | hovering | yolo26s_1024 | raw_diff | <=0 | — | 215 / 15 |
| H2 | iso_L3 | 0.02 | >20 | hovering | yolo26s_1024 | raw_diff | <=0 | — | 310 / 12 |
| H2 | iso_L5 | 0.02 | >20 | hovering | yolo26s_1024 | raw_diff | <=0 | — | 310 / 12 |
| H2 | iso_L3 | 0.02 | 6-20 | moving | yolo26s_1024 | raw_diff | <=0 | — | 167 / 10 |
| H2 | iso_L5 | 0.02 | 6-20 | moving | yolo26s_1024 | raw_diff | <=0 | — | 167 / 10 |
| H2 | iso_L3 | 0.02 | 6-20 | hovering | yolo26s_1024 | tiny_det | <=0 | — | 215 / 15 |
| H2 | iso_L5 | 0.02 | 6-20 | hovering | yolo26s_1024 | tiny_det | <=0 | — | 215 / 15 |
| H2 | iso_L3 | 0.02 | >20 | hovering | yolo26s_1024 | tiny_det | <=0 | — | 310 / 12 |
| H2 | iso_L5 | 0.02 | >20 | hovering | yolo26s_1024 | tiny_det | <=0 | — | 310 / 12 |
| H2 | iso_L3 | 0.02 | 6-20 | moving | yolo26s_1024 | tiny_det | <=0 | — | 167 / 10 |
| H2 | iso_L5 | 0.02 | 6-20 | moving | yolo26s_1024 | tiny_det | <=0 | — | 167 / 10 |
| H3/H5 | 3 | — | 6-20 | hovering | yolo26s_1024 | border_band | + | +0.0401 | 215 / 15 |
| H3/H5 | 5 | — | 6-20 | hovering | yolo26s_1024 | border_band | <=0 | -0.0306 | 215 / 15 |
| H3/H5 | 3 | — | >20 | hovering | yolo26s_1024 | border_band | + | +0.0402 | 310 / 12 |
| H3/H5 | 5 | — | >20 | hovering | yolo26s_1024 | border_band | <=0 | -0.0301 | 310 / 12 |
| H3/H5 | 3 | — | 6-20 | moving | yolo26s_1024 | border_band | <=0 | -0.3231 | 167 / 10 |
| H3/H5 | 5 | — | 6-20 | moving | yolo26s_1024 | border_band | <=0 | -0.3760 | 167 / 10 |
| H3/H5 | 3 | — | 6-20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | -0.2413 | 215 / 15 |
| H3/H5 | 5 | — | 6-20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | -0.3094 | 215 / 15 |
| H3/H5 | 3 | — | >20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | -0.2448 | 310 / 12 |
| H3/H5 | 5 | — | >20 | hovering | yolo26s_1024 | orb_lite_diff | <=0 | -0.3131 | 310 / 12 |
| H3/H5 | 3 | — | 6-20 | moving | yolo26s_1024 | orb_lite_diff | <=0 | -0.4366 | 167 / 10 |
| H3/H5 | 5 | — | 6-20 | moving | yolo26s_1024 | orb_lite_diff | <=0 | -0.3116 | 167 / 10 |
| H3/H5 | 3 | — | 6-20 | hovering | yolo26s_1024 | raw_diff | <=0 | -0.0407 | 215 / 15 |
| H3/H5 | 5 | — | 6-20 | hovering | yolo26s_1024 | raw_diff | <=0 | -0.1291 | 215 / 15 |
| H3/H5 | 3 | — | >20 | hovering | yolo26s_1024 | raw_diff | <=0 | -0.0428 | 310 / 12 |
| H3/H5 | 5 | — | >20 | hovering | yolo26s_1024 | raw_diff | <=0 | -0.1313 | 310 / 12 |
| H3/H5 | 3 | — | 6-20 | moving | yolo26s_1024 | raw_diff | <=0 | -0.2682 | 167 / 10 |
| H3/H5 | 5 | — | 6-20 | moving | yolo26s_1024 | raw_diff | <=0 | -0.1438 | 167 / 10 |
| H3/H5 | 3 | — | 6-20 | hovering | yolo26s_1024 | tiny_det | <=0 | -0.4310 | 215 / 15 |
| H3/H5 | 5 | — | 6-20 | hovering | yolo26s_1024 | tiny_det | <=0 | -0.3885 | 215 / 15 |
| H3/H5 | 3 | — | >20 | hovering | yolo26s_1024 | tiny_det | <=0 | -0.5065 | 310 / 12 |
| H3/H5 | 5 | — | >20 | hovering | yolo26s_1024 | tiny_det | <=0 | -0.3908 | 310 / 12 |
| H3/H5 | 3 | — | 6-20 | moving | yolo26s_1024 | tiny_det | <=0 | -0.3752 | 167 / 10 |
| H3/H5 | 5 | — | 6-20 | moving | yolo26s_1024 | tiny_det | <=0 | -0.3962 | 167 / 10 |
| H8 | H8_L3 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | border_band | <=0 | +0.0020 | 215 / 15 |
| H8 | H8_L5 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | border_band | + | +0.0112 | 215 / 15 |
| H8 | H8_L3 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | border_band | <=0 | +0.0020 | 310 / 12 |
| H8 | H8_L5 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | border_band | + | +0.0112 | 310 / 12 |
| H8 | H8_L3 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | border_band | + | +0.0281 | 167 / 10 |
| H8 | H8_L5 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | border_band | + | +0.0282 | 167 / 10 |
| H8 | H8_L3 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | orb_lite_diff | + | +0.0254 | 215 / 15 |
| H8 | H8_L5 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | orb_lite_diff | + | +0.0251 | 215 / 15 |
| H8 | H8_L3 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | orb_lite_diff | + | +0.0258 | 310 / 12 |
| H8 | H8_L5 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | orb_lite_diff | + | +0.0256 | 310 / 12 |
| H8 | H8_L3 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | orb_lite_diff | + | +0.0298 | 167 / 10 |
| H8 | H8_L5 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | orb_lite_diff | <=0 | -0.0042 | 167 / 10 |
| H8 | H8_L3 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | raw_diff | + | +0.0210 | 215 / 15 |
| H8 | H8_L5 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | raw_diff | + | +0.0285 | 215 / 15 |
| H8 | H8_L3 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | raw_diff | + | +0.0214 | 310 / 12 |
| H8 | H8_L5 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | raw_diff | + | +0.0290 | 310 / 12 |
| H8 | H8_L3 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | raw_diff | + | +0.0080 | 167 / 10 |
| H8 | H8_L5 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | raw_diff | <=0 | -0.0160 | 167 / 10 |
| H8 | H8_L3 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | tiny_det | + | +0.0365 | 215 / 15 |
| H8 | H8_L5 | — | 6-20 | hovering | yolo26s_1024>yolo26n_640 | tiny_det | + | +0.0162 | 215 / 15 |
| H8 | H8_L3 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | tiny_det | + | +0.0342 | 310 / 12 |
| H8 | H8_L5 | — | >20 | hovering | yolo26s_1024>yolo26n_640 | tiny_det | <=0 | +0.0003 | 310 / 12 |
| H8 | H8_L3 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | tiny_det | + | +0.0326 | 167 / 10 |
| H8 | H8_L5 | — | 6-20 | moving | yolo26s_1024>yolo26n_640 | tiny_det | + | +0.0218 | 167 / 10 |

Danh sách đầy đủ (cả ô mô tả) + Δ dự đoán: prereg_cells.json (cùng thư mục kết quả).
