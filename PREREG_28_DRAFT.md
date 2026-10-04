# PREREG_28 — DRAFT (NOT frozen)

> **Bản nháp — KHÔNG hash, KHÔNG git tag.** Chỉ đóng băng (sha256 + tag) sau khi đo $q_{in}, q_{out}, q_b, q_o, c$ trên tập TRAIN ở P2 và thay các ký hiệu dưới đây bằng số. Công thức: THEORY_28.md (§5–§6). Số P0 dùng làm đầu vào: results/p0_stats.json (`p0b`).

## 1. Quantity being predicted

For each cell, $\Delta = \mathrm{miss}_{periodic} - \mathrm{miss}_{gate}$ at **equal total cost** (detector activation + cue cost $c$), and the iso-KPI cost ratio $a_{gate}/a_{periodic}$ at a target KPI. Positive $\Delta$ = the gate wins. Two KPIs:

- **Miss KPI** (event recall): KPI window = whole first visible run, $T_e = D_e$.
- **Latency KPI** (new-vehicle onset latency): $T_e = \min(D_e, L_{max}+1)$, $L_{max} \in \{3, 5, 10\}$ frames (UAVDT: 0.10 / 0.17 / 0.33 s).

Policies (all replayed offline on the same detector dump): periodic with uniform phase; gate ∨ refresh$(R)$ with cues {raw frame-diff, ORB-compensated diff, border-entry cue, tiny detector yolo26n@320}; pure gate ($R = \infty$); per-sequence oracle stride (ceiling).

## 2. Cells, operating point and confirmatory family (locked 28/09/2026, P2C — before any real channel is measured)

**Operating point rule (B1).** Each cue gets exactly ONE threshold $\theta^*$, chosen on TRAIN only by a fixed rule:
$\theta^* = \arg\max_\theta\,[q_{in}(\theta) - q_{out}(\theta)]$ with the onset window $w = 5$ (Youden index on the onset channel), subject to
$a_G(\theta^*) = 1/R + (1-1/R)\,[\rho_5 q_{in} + (1-\rho_5) q_{out}] + c \le 0.25$ at $R = 30$; ties are broken by the smaller $a_G$.
$q_{out}$ here is the measured out-of-window firing rate on TRAIN (i.e. $q_{out,eff}$ averaged over the TRAIN density). This rule is fixed before any measurement;
code: `p2_prereg_build.theta_star()` (the channel estimator always fits $(q_b, q_o)$ at this $\theta^*$).

**Axes.**

| Axis | Levels | In confirmatory family |
|---|---|---|
| dataset | UAVDT (TEST split); VisDrone2019-VID (val + test-dev) if the data arrive — reported separately | both |
| cue | raw frame-diff, ORB-compensated diff, border-entry cue, tiny detector yolo26n@320 — each at its $\theta^*$ | all 4 |
| detector level | yolo26s@1024 (main); yolo26n@1024 and yolo26s@640 (lower recall) | main for H2/H3/H5; pairs (s@1024, n@1024), (s@1024, s@640) for H8 |
| policy | gate ∨ refresh with $R = 30$ (main policy); pure gate $R = \infty$ | $R = 30$ only ($R = \infty$ descriptive) |
| onset window | $w = 5$ | fixed |
| KPI | latency $L_{max} \in \{3, 5, 10\}$; miss; $L_{max} = 30$ | latency 3/5/10 |
| iso-KPI target | $\varepsilon \in \{1\%, 2\%, 5\%\}$ | 2 %, 5 % |
| density | $M$ bin at onset: 1–5, 6–20, > 20 | all bins that pass the size rule |
| ego-motion | hovering / moving (threshold from the P0 shift distribution after frames arrive) | both |
| ROI margin $m$ | 0 (main), 5 % | 0 only (5 % = sensitivity) |

**Size rule.** A cell needs ≥ 30 events and ≥ 3 sequences (counted on TRAIN when building; re-checked on TEST when scoring). On UAVDT TRAIN the bin $M$ 1–5 has 26 events in 5 sequences → excluded; 6–20 (382 / 25) and > 20 (317 / 14) pass.

**Reduction rule (target 30–80 cells).** If the family exceeds 80 cells, axes are dropped in this fixed order, one step at a time, until ≤ 80. The order depends only on cell counts, never on outcomes:
1. $\varepsilon = 5\%$ (keep 2 %); 2. $L_{max} = 10$; 3. H8 pooled over $M$ bins; 4. H8 at $L_{max} = 3$ only; 5. H8 pooled over ego; 6. iso-KPI at $L_{max} = 3$ only; 7. matched cost at $L_{max} = 3$ only.
Steps 1–2 were set in the P2C prompt; steps 3–7 were added by Claude (P2C) because steps 1–2 alone leave 128 cells when both ego classes pass — **to be approved at freeze**.

**Estimated family size (UAVDT only, from TRAIN counts; ego labels not yet available).** Per stratum (cue × $M$ bin × ego): 3 matched-cost + 6 iso-KPI + 6 H8 cells.
Without an ego split: 4 × 2 × 1 strata = 120 → drop $\varepsilon = 5\%$: 96 → drop $L_{max} = 10$: **64**.
With both ego classes passing the size rule: 240 → 192 → 128 → H8 pooled over $M$: 96 → H8 at $L_{max}=3$: **80**.
If VisDrone arrives, the same order continues on the combined family.

## 3. Predicted sign of Δ (functional form; numbers filled in at freeze)

Notation: $\rho_w = \rho_{onset}(w)$ measured per cell (P0b-A5), $\rho_1 \approx$ onset rate per frame, $q_{out,eff}(M) = 1 - (1-q_b)(1-q_o)^{M}$, $\kappa(R,T) = (R-T)/(T(R-1))$.

| # | Cell family | Prediction | Rule (THEORY_28) |
|---|---|---|---|
| H1 (descriptive) | Miss KPI, any cue, $a \ge 0.05$ | $\Delta \le 0$ (periodic not beaten) except where $\mathbb{E}_D[\mathcal G - \mathcal C - Tc] > 0$ | Thm G3 averaged over $F_D$; P0b-A3: periodic already reaches miss ≤ 5 % at $S^* = 20$ ($a = 0.05$) for $r = 1$ on UAVDT E3 |
| H2 | Latency KPI, iso-KPI at $\varepsilon$ | gate cheaper **iff** $\rho_1 + (1-\rho_1)\,q_{out,eff}(M) + c < (1-\varepsilon)/T$ (for $r<1$: $\rho_1$ replaced by the activation needed for $n(r,\varepsilon)$ onset looks) | Cor. G5, Thm P3(iii) |
| H3 | Latency KPI, matched cost, $M > M^*$ (per cell) | $\Delta \le 0$ | Thm P3(i); $M^*$ computed from TRAIN estimates |
| H4 | moving drone, uncompensated cue (raw diff) | $\Delta \le 0$ in every cell | Thm P3(ii) ($q_b \to 1$) |
| H5 | compensated cue with cost $c$ | $\Delta \le 0$ whenever $(1-\rho)q_{out,eff} + c/(1-1/R) \ge \kappa(R,T)$ | Cor. G4 (N-G) |
| H6 (descriptive) | $L_{max} = 30$ (T = 31) | gate never cheaper at iso-KPI (UAVDT: $(1-\varepsilon)/31 \le 0.032 < \rho_1 \approx 0.042$) | Cor. G5 |
| H8 (P2B) | Latency KPI, matched cost, pairs of detector levels (yolo26n vs yolo26s; 640 vs 1280) | $\Delta$ is **larger for the lower-recall detector** when both recalls exceed $r^\dagger$ of the cell's cue configuration; otherwise no prediction of increase | THEORY Remark G2′ (conditional: Δ(r) concave, Δ(0)=0; false below $r^\dagger$). Simulated check: sign predicted correctly in 97.5 % of 720 pairs |
| H7 | $m = 5\%$ vs $m = 0$ | the gate's advantage shrinks (fewer short border events; ROI $D<8$ tail 4.1 % → 2.5 %) | P0b-A2 |

**Expected location of the boundary (to be confirmed at freeze):** gate wins only in the corner {latency KPI with $L_{max} \le 5$, sparse traffic ($M$ 1–5 or 6–20 with small $q_o$), hovering or compensated with small $c$, border-entry cue}. Everywhere else the periodic schedule is at least as good.

*Superseded at freeze by the PREREG_28 table (real TRAIN channel, 01-10-2026).*

## 4. Confirmatory family, scoring and refutation criteria (locked 27/09/2026, P2B)

**Family.** Confirmatory = latency KPI only ($L_{max} \in \{3, 5, 10\}$) at matched cost (H3/H5), the iso-KPI boundary cells H2 at $\varepsilon \in \{2\%, 5\%\}$, and H8; each cell needs ≥ 30 events and ≥ 3 sequences; ROI $m = 0$ is primary, $m = 5\%$ a sensitivity analysis. Descriptive only (not counted): miss KPI (H1), $L_{max} = 30$ (H6), iso-KPI at $\varepsilon = 1\%$ (appendix — on UAVDT TRAIN $\varepsilon = 1\%$ is set by the 1.8 % of events with $D = 1$, see results/p2/sim/tail_analysis.json). The builder applies §2 (one $	heta^*$ per cue, $R = 30$, main detector, reduction order) and prints the family size and the reduction log (P2B had 5 760 simulated cells; after P2C: 64 on the simulated channel, 64–80 estimated for UAVDT).

~~**Scoring (per confirmatory cell).** A prediction "$\le 0$" is correct iff the 95 % paired bootstrap CI of Δ contains 0 or lies below 0. A prediction "$> 0$" is correct iff the point estimate is > 0 and the CI does not lie entirely below 0. Iso-KPI cells: correct iff "gate cheaper at equal KPI" matches. Holm is applied within the confirmatory family only to **report** how many cells are significant after correction; it does not define correctness.~~ *(P2B rule, replaced 01-10-2026 by Q6 below.)*

**Scoring (per confirmatory cell) — three-way, P2H-Q6 (01-10-2026).** CI = 95 % paired bootstrap by sequence (B = 1000, seed 42); $\delta_{min} = 0.005$ (= the builder's MARGIN).
- Prediction "+" ($\Delta_{pred} > \delta_{min}$): CORRECT iff CI$_{lo}(\hat\Delta) > 0$; WRONG iff CI$_{hi} \le 0$; otherwise INCONCLUSIVE.
- Prediction "≤0": CORRECT iff CI$_{hi} < \delta_{min}$; WRONG iff CI$_{lo} > \delta_{min}$; otherwise INCONCLUSIVE.
- Iso-KPI (H2): CORRECT/WRONG as before ("gate cheaper at equal KPI" matches the prediction); INCONCLUSIVE when the CI of $a_G - a_P(arepsilon)$ contains 0.
- H8: the same "+"/"≤0" rule applied to $\widehat{\Delta\Delta}$. H8 cells whose configuration has $r^\dagger \ge r_{main}$ or $r^\dagger \ge r_{low}$ get prediction "—" and leave the confirmatory family (Q7; on TRAIN 01-10-2026 none did).
- Holm is applied within the confirmatory family only to report how many cells are significant; it does not define correctness. Code: `code/p2/p2_prereg_score.py`.

**Refutation criteria** (the preregistered claims are rejected if any holds):
1. ~~fewer than 80 % of confirmatory cells are correct~~ → (Q6) fewer than 80 % of the confirmatory cells that are DECIDED (correct or wrong) are correct;
2. the observed iso-KPI boundary $q_o^*$ (THEORY Thm P3-iii; P2B-A1) lies outside its predicted bootstrap CI in ≥ 2/3 of the density groups;
3. the sign of H8 ("Δ(latency) increases as recall decreases") is wrong in ≥ 50 % of detector pairs.

**Power criterion P (Q6).** If more than 50 % of the confirmatory cells are INCONCLUSIVE, the paper's conclusion is "not testable with these data" (neither confirmed nor rejected); the three-way shares are reported separately for H2, H3–H5 and H8. P is checked before criteria 1–3.
Criterion 3 is evaluated on decided H8 cells. Pipeline check with the new rule on the simulated P2C channel: see results/p2/sim/prereg_sim_score.json (`three_way`, `three_way_by_group`).

Statistics: paired bootstrap by sequence, B = 1000, seed = 42; p for Holm = two-sided normal approximation from the bootstrap SE.

**Pipeline check on the simulated channel (not evidence):** P2B (5 760 cells, 40 operating points): 98.9 % correct. P2C (reduced family: 4 pseudo-cues = ROC paths through the A7 grid at $w = 5$, $	heta^*$ chosen by the B1 rule, $R = 30$, main $r = 1$, H8 pairs $(1, 0.8), (1, 0.5)$): **64 cells, 98.4 % correct (63/64)**; H2 15/16, H3/H5 16/16, H8 32/32; criterion 2: 0/20 groups outside; criterion 3: 0 % wrong (results/p2/sim/prereg_sim_score.json, PREREG_SIM_TEST.md). Near-perfect agreement is expected because the simulated channel obeys the model exactly; it only shows that builder, family rules and scorer are wired consistently.

## 5. What is still open before freezing

1. $q_{in}, q_{out}$ per cue and per detector level on TRAIN (P2); $q_b$ per ego-motion class; $q_o$ from the regression of $\ln(1-q_{out})$ on $M$.
2. Cue cost $c$ from `results/p2/bench_openvino.json` (laptop, OpenVINO); remeasure on real frames.
3. Hovering threshold from the P0 ego-motion distribution (needs UAVDT / VisDrone frames).
4. VisDrone-VID data (download blocked; see DOWNLOAD_BLOCKED.md).
5. ~~Final list of confirmatory cells: one operating point per cue~~ → rule fixed in §2 (P2C); the list itself is generated by the builder from the TRAIN channel.
6. $r^\dagger$ per cue configuration for H8, from the TRAIN channel and measured detector recalls.

## 6. Unknown before freezing (P2C, 28/09/2026)

The rules above are fixed; the following numbers are not yet known and will be filled in from TRAIN only:

| Unknown | Needed for | Source | Status 28/09 |
|---|---|---|---|
| real $q_{in}(\theta), q_{out}(\theta)$ per cue, hence $\theta^*$ | every prediction | `p2_cue_trace.py` + `p2_estimate_channel.py` on UAVDT TRAIN frames | waiting for UAVDT frames (`UAV-benchmark-M.zip`) |
| $q_b$ per ego class, $q_o$ | $q_{out,eff}(M)$, H3/H4, boundary $q_o^*$ | same | same |
| real cue cost $c$ | $a_G$, H5, $\theta^*$ constraint | cue trace $t_{ms}$ / detector $t_{ms}$ (yolo26s@1024, VisDrone weights) | detector timing: C1 bench with VisDrone weights; cue timing needs frames |
| detector recalls $r$ for the 3 levels | $r^\dagger$ (H8), all $r$-dependent predictions | TRAIN dumps (yolo26s@1024, yolo26n@1024, yolo26s@640) | yolo26s weights verified (re-exported no-NMS); yolo26n weights not yet trained |
| hovering threshold | ego axis, size rule per ego | `p0_egomotion.py` + `p0b_hover_threshold.py` | needs frames; two candidate thresholds will be reported |
| whether VisDrone-VID is available | dataset axis, family size | manual download (DOWNLOAD_BLOCKED.md) | not available |
| final family size (64–80 for UAVDT) | refutation criterion 1 denominator | builder log | depends on ego labels |


## Quyết định 01-10-2026 (anh Đạt, P2F) — trước freeze, trước khi mở TEST

Các luật dưới đây cố định; số đo tương ứng do `code/p2/p2_prereg_build.py` sinh vào PREREG_28.md (§0–§6), không gõ tay.

- **Q1 — Ngưỡng score "phát hiện"** (thay quy ước 0,05 của P2E): **chính = 0,25** (mặc định Ultralytics predict, điểm vận hành). 0,05 và 0,50 = phụ lục mô tả. `p2_detector_recall` / `p2_estimate_channel`: score_min_main = 0,25, r_levels tính lại. Cue tiny_det giữ θ* riêng. Kênh v1 (score 0,05) giữ ở results/p2/channel_train_score005.json cho phụ lục.
- **Q2 — Luật H8** (PREREG §2): mức recall thấp hợp lệ khi r_main − r_low ≥ 0,05 (score 0,25, TRAIN, ước lượng điểm, IoU 0,5). Ứng viên: yolo26n@1024, yolo26s@640, yolo26n@640. Cặp H8 = (yolo26s@1024, mức hợp lệ có r thấp nhất), tối đa 2 cặp (cài đặt: các mức hợp lệ xếp theo r tăng dần, lấy tối đa 2). Không mức nào hợp lệ → H8 ra khỏi họ xác nhận (thành mô tả). Không train thêm detector.
- **Q3 — Dữ liệu** (PREREG §1): M0207 chỉ dùng khung ≤ khung GT cuối (571); M0901 tính biên theo bề rộng thật 960. Áp cho cả TEST khi mở. P0 tái sinh v3 (bản v2 giữ: P0_REPORT_28_v2_bak.md); numbers.tex tái sinh, tên macro không đổi.
- ~~**Q4 — A6 audit tự động khung trước sinh** (PREREG §5):~~ **RÚT 01-10-2026 chiều (Q5, xem dưới)** — luật cũ: `code/p2/p2_audit_prebirth.py`. Mỗi sự kiện E3: box GT ở khung sinh b; dump yolo26s@1024 (score ≥ 0,25) ở b−5…b−1; khớp = IoU ≥ 0,3 với box GT khung b. ≥ 3/5 khớp → annotation-late; ≤ 2/5 → true-birth; b−5 < 1 → censored-start. **annotation-late bị LOẠI khỏi E3 ở cả hai split**; báo số loại trên TRAIN; > 20 % interior bị loại → hạn chế dữ liệu (§6). Kiểm phụ trên crop = model-assisted, không phải nhãn người; bất đồng không đổi nhãn tự động. TEST không mở; chạy audit TEST sau freeze với cùng luật.
- **Luật chi phí** (P2F, ghi tường minh): cue có c ≥ 1 bị loại trước khi chọn θ* (Thm G3) — ego_comp_diff (c ≈ 1,02 trên iGPU) bị loại theo luật này, không phải vì kết quả.
- **Luật cột "dự đoán"**: "+" iff Δ_pred > δ_min với δ_min = MARGIN = 0,005 (matched cost và H8 ΔΔ); iso-KPI: "+" iff miss_G ≤ ε và a_G < a_P(ε).
- **Sửa lỗi P2F**: builder đọc nhãn ego từ cột `ego` của egomotion_seq.parquet (ngưỡng tạm 1,0 px → 24/6 chuỗi) thay vì ngưỡng phân bố 0,61 px (19/11) mà kênh dùng; đã sửa để builder và kênh cùng một nhãn.

## Quyết định 01-10-2026 chiều (anh Đạt, P2G) — Q5 rút luật loại "annotation-late"

- **Q5**: RÚT luật loại annotation-late của Q4. Audit giữ lại nhưng **chỉ mô tả, không loại sự kiện** (cả TRAIN và TEST). Tập E3 = định nghĩa P0 (VISIBLE: occlusion ≠ 2 và out_of_view ≠ 2), đầy đủ. Lý do: luật cũ gộp (i) xe vào từ mép lộ một phần trước khi đủ hiện và (ii) track đã có trong GT ở khung trước với occlusion = 2 / out_of_view = 2 — tức chính định nghĩa onset của P0 — thành "gán trễ".
- **Luật audit mới (PREREG §5, cố định, áp cho TEST sau freeze)**: b = khung onset (khung VISIBLE đầu tiên); detector yolo26s@1024 score ≥ 0,25 ở b−5…b−1, khớp IoU ≥ 0,3 với box GT khung b. b−5 < khung đầu → censored-start; ≤ 2/5 khớp → true-birth; ≥ 3/5 khớp: A1 partial-entry = border và (box khung b cách mép ≤ 2 px hoặc GT trước đó out_of_view = 2); A2 visibility-transition = GT trước đó occlusion = 2 (hoặc out_of_view = 2 khi interior); A3 annotation-late THẬT = không có dòng GT nào của track trong b−5…b−1; A4 còn lại → true-birth. (Cài đặt: b = start thay vì khung GT đầu tiên như P2F — với khung GT đầu, track không thể có GT trước b nên A2 không xảy ra.)
- Chỉ khi A3 > 20 % interior trên TRAIN → ghi hạn chế ở PREREG §6, vẫn không loại. Phụ lục mô tả: onset_alt = khung sớm nhất trong b−5…b−1 có detector khớp; KPI độ trễ ô chính tính lại với onset_alt, báo cạnh onset GT.
- P0 v3 gộp hai split chỉ dùng thống kê dữ liệu thô (không lọc, không audit); audit và mọi ước lượng chỉ TRAIN.
- **Bản 01-10 sáng (P2F, lọc 82 % E3 TRAIN) bị rút vì luật audit sai; không số nào từ bản đó được dùng.**
- Queue dump còn lại (s@960, s@1280, n@960, n@1280) chỉ phục vụ mô tả, không chặn freeze; chạy đêm sau khi pause Google Drive, `code/p2/p2_dump_queue_hidden.ps1` (Start-Process -WindowStyle Hidden, log ra file).
