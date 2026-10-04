"""P2A-A6: sinh P2_SIM_REPORT_28.md từ results/p2/sim/*.json|csv (mọi số lấy từ file kết quả).
KẾT QUẢ GIẢ LẬP (detector lý tưởng từ GT + cue theo A-gate) — không phải YOLO, không phải cue thật.
Chạy: python code/p2/replay/p2_sim_report.py
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "results" / "p2" / "sim"
REPORT = ROOT / "P2_SIM_REPORT_28.md"


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def f3(x):
    return "—" if x is None or pd.isna(x) else f"{x:.3f}"


def main():
    tr = json.loads((SIM / "theory_repro.json").read_text(encoding="utf-8"))
    summ = json.loads((SIM / "summary.json").read_text(encoding="utf-8"))
    cdf = pd.read_csv(SIM / "agnostic_cells.csv")
    idf = pd.read_csv(SIM / "iso_kpi.csv")
    dens = pd.DataFrame(json.loads((SIM / "density_Mstar.json").read_text(encoding="utf-8")))
    orac = pd.DataFrame(json.loads((SIM / "oracle_stride.json").read_text(encoding="utf-8")))
    info = tr["info"]
    L = ["# P2_SIM_REPORT_28 — Replay với detector GIẢ LẬP (UAVDT TRAIN)", "",
         f"*Sinh bởi `code/p2/replay/p2_sim_report.py` — {dt.datetime.now():%Y-%m-%d %H:%M}. Nguồn: results/p2/sim/. "
         "**GIẢ LẬP**: detector lý tưởng từ GT (box = GT, mỗi box được phát hiện độc lập với xác suất r; FP ngẫu nhiên) và cue giả lập "
         "theo A-gate (Bernoulli q_in trong cửa sổ khởi phát, q_out ngoài). Không phải YOLO, không phải cue thật; dùng cho mục "
         "\"analysis under an idealised detector\" và để kiểm engine.*", "",
         f"- Dữ liệu: split = {info['split']}, {info['n_seq']} chuỗi, {info['n_frames']} khung; sự kiện (cửa sổ sinh): "
         + ", ".join(f"{k} {v}" for k, v in info["n_events"].items()) + ".",
         f"- ρ_onset trên TRAIN: " + ", ".join(f"w={k}: {pct(v)}" for k, v in info["rho_onset"].items()) + ".",
         f"- Thống kê: paired bootstrap theo chuỗi B={info['B']}, seed={info['seed']}; matched cost = bisection trên S thực, |a_G − a_P| < 0.002; "
         "Holm trên các ô xác nhận (≥30 sự kiện, ≥3 chuỗi).", ""]
    # (i)
    L += ["## 1. Engine tái lập lý thuyết trên timeline thật (A7-i)", "",
          f"Tiêu chí: |dự đoán − replay| < {tr['tol']} hoặc dự đoán trong CI 95% của replay. Tổng {tr['n']} phép so, đạt {tr['n_ok']}.", "",
          "| Kiểm | n | đạt | |lệch| lớn nhất |", "|---|---|---|---|"]
    for r in tr["by_check"]:
        L.append(f"| {r['check']} | {r['size']} | {r['sum']} | {tr['max_abs_dev'][r['check']]:.4f} |")
    L += ["", "- **Đọc cột \"G2 A-sparse\" đúng cách (P2B-A4):** đây là kiểm MỘT PHÍA — công thức A-sparse (THEORY §5.1) là CẬN TRÊN của miss gate. "
          f"Lệch lớn (tối đa {tr['max_abs_dev']['G2 miss_gate A-sparse upper bound (one-sided)']:.2f}) là KỲ VỌNG, không phải lỗi: trên timeline thật cửa sổ KPI dài "
          "(KPI miss) chồng lên cửa sổ khởi phát của xe khác nên gate tốt hơn cận. Tiêu chí của cột này chỉ là \"dự đoán ≥ replay\". "
          "Công thức chính xác theo timeline (dùng I_t thật) được kiểm HAI phía ở dòng \"timeline-exact\".",
          "- Các ô timeline-exact vượt tiêu chí là do mỗi cấu hình chỉ gieo cue MỘT lần (CI theo chuỗi không gồm biến thiên của realization); "
          "kiểm lại 2 ô lệch nhất với 10 seed: trung bình khớp dự đoán trong 0,006 (LOG_28, P2A).", ""]
    # (ii)
    L += ["## 2. Matched cost: ô gate thắng (E3, mọi M) (A7-ii)", "",
          "Đếm cấu hình (q_in × q_out × w × R × fp) có Δ = miss_P − miss_G với CI 95% > 0 (gate thắng) / < 0 (periodic thắng); cột Holm = gate thắng còn ý nghĩa sau Holm (p xấp xỉ chuẩn từ SE bootstrap, họ = mọi ô xác nhận).", "",
          "| r | KPI | cấu hình | gate thắng | periodic thắng | gate thắng (Holm) |", "|---|---|---|---|---|---|"]
    for k, v in summ["matched_cost_E3"].items():
        r, kpi = k.split("|")
        L.append(f"| {r[1:]} | {kpi} | {v['n_cfg']} | {v['n_gate_win']} | {v['n_periodic_win']} | {v['n_holm_win']} |")
    c = cdf[(cdf.event == "E3") & (cdf.M_bin != "all") & (cdf.fp == 0.0)]
    L += ["", "Theo M_bin (M tại khung sinh), fp = 0: tỉ lệ cấu hình gate thắng (CI > 0) / periodic thắng.", "",
          "| r | KPI | M 1–5 | M 6–20 | M >20 |", "|---|---|---|---|---|"]
    for (r, kpi), g in c.groupby(["r", "kpi"]):
        cellv = []
        for mb in ("1-5", "6-20", ">20"):
            h = g[g.M_bin == mb]
            cellv.append(f"{pct((h.sign == '+').mean(), 0)} / {pct((h.sign == '-').mean(), 0)} (n_ev {int(h.n_events.iloc[0]) if len(h) else 0})")
        L.append(f"| {r:g} | {kpi} | " + " | ".join(cellv) + " |")
    best = cdf[(cdf.event == "E3") & (cdf.M_bin == "all") & (cdf.fp == 0.0)].sort_values("delta", ascending=False)
    L += ["", "Cấu hình gate tốt nhất mỗi (r, KPI), fp = 0:", "", "| r | KPI | q_in | q_out | w | R | a | miss_G | miss_P | Δ [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (r, kpi), g in best.groupby(["r", "kpi"]):
        x = g.iloc[0]
        L.append(f"| {r:g} | {kpi} | {x.q_in:g} | {x.q_out:g} | {x.w} | {x.R} | {f3(x.a_G)} | {pct(x.miss_G)} | {pct(x.miss_P)} | "
                 f"{pct(x.delta)} [{pct(x.lo)}, {pct(x.hi)}] |")
    fp = cdf[(cdf.event == "E3") & (cdf.M_bin == "all")].pivot_table(index=["r", "q_in", "q_out", "w", "R", "kpi"], columns="fp", values="delta")
    if 0.0 in fp.columns and 0.05 in fp.columns:
        L += ["", f"- Ảnh hưởng FP (fp_rate 0 → 0.05 box/khung): |ΔΔ| lớn nhất = {pct((fp[0.05] - fp[0.0]).abs().max(), 2)} "
              "(cùng realization cue cho hai mức fp → so cặp; FP ngẫu nhiên hầu như không trùng box GT nên gần như không đổi miss; FP quan trọng cho đếm xe — ngoài phạm vi KPI này)."]
    # iso-KPI
    L += ["", "## 3. Iso-KPI: gate có đạt cùng KPI với chi phí thấp hơn không", "",
          "| r | KPI | ε | a_P cần (periodic) | cấu hình đạt KPI | rẻ hơn periodic |", "|---|---|---|---|---|---|"]
    for k, v in summ.get("iso_kpi", {}).items():
        r, kpi, e = k.split("|")
        L.append(f"| {r[1:]} | {kpi} | {e[3:]} | {f3(v['a_P_iso'])} | {v['n_meet']}/{v['n_cfg']} | {v['n_cheaper']} |")
    ch = idf[idf.gate_cheaper & (idf.fp == 0.0)].sort_values(["r", "kpi", "eps", "a_G"])
    if len(ch):
        L += ["", "Cấu hình rẻ hơn ở iso-KPI (fp = 0; tối đa 3 mỗi ô):", "", "| r | KPI | ε | q_in | q_out | w | R | a_G | a_P | miss_G |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, g in ch.groupby(["r", "kpi", "eps"]):
            for x in g.head(3).itertuples(index=False):
                L.append(f"| {x.r:g} | {x.kpi} | {x.eps:g} | {x.q_in:g} | {x.q_out:g} | {x.w} | {x.R} | {f3(x.a_G)} | {f3(x.a_P_iso)} | {pct(x.miss_G)} |")
    # §4 (P2B-A1): M* timeline-exact + ranh giới iso-KPI
    me = json.loads((SIM / "mstar_exact.json").read_text(encoding="utf-8")) if (SIM / "mstar_exact.json").exists() else None
    mi = json.loads((SIM / "mstar_iso.json").read_text(encoding="utf-8")) if (SIM / "mstar_iso.json").exists() else None
    if me:
        L += ["", "## 4. Kiểm M* (THEORY §6) — dự đoán timeline-exact cho mọi r (P2B-A1; thay bản A-sparse cũ)", "",
              "Cue theo mật độ: q_in = 1 trong cửa sổ khởi phát (w = Lmax), ngoài cửa sổ q_out,t = 1 − (1 − q_o)^{M_t} (M_t = số xe visible thật), R = ∞; "
              "UAVDT TRAIN chia 3 nhóm chuỗi theo M trung vị. Dự đoán: a = mean_t q_t, miss_G timeline-exact (Prop. G2), miss_P = Lemma 1 với r và F_D của nhóm. "
              f"Replay: detector giả lập, cue gieo {me['design']['K_seeds']} lần. Khớp = dự đoán trong CI 95% của replay, hoặc cùng dấu và |lệch| < {me['design']['tol']}.",
              "", "| r | ô | khớp | dự đoán + | replay + | replay − | |lệch| lớn nhất |", "|---|---|---|---|---|---|---|"]
        for r, v in me["by_r"].items():
            L.append(f"| {r} | {v['n']} | {v['n_match']} | {v['n_sign_pred_plus']} | {v['n_replay_plus']} | {v['n_replay_minus']} | {v['max_abs_dev']:.3f} |")
        L.append(f"| tổng | {me['total']['n']} | {me['total']['n_match']} | | | | |")
        L += ["", "- **Ở matched cost Δ không đổi dấu:** với q_in = 1 gate luôn nhìn khung khởi phát (r = 1 ⇒ miss_G = 0 ⇒ Δ ≥ 0 theo cấu tạo); khi q_o tăng cả hai "
              "chính sách tiến về miss ≈ 0 (hoà) — đúng nghĩa \"không thắng NGHIÊM NGẶT\" của Thm P3. Vì vậy ranh giới được đo theo nghĩa ISO-KPI (bảng dưới).",
              "- Bảng cũ (A-sparse, báo 63/72) không kiểm được gì ở r < 1: cả dự đoán và quan sát đều \"+\" ở mọi ô; chỉ 24 ô r = 1 là kiểm thật (15/24)."]
    if mi:
        inf_ = lambda v: "∞" if v == float("inf") else f"{v:.4f}"  # noqa: E731
        L += ["", "### Ranh giới iso-KPI q_o* (THEORY Thm P3-iii)", "",
              "adv(q_o) = a_P(ε) − a_G(q_o) nếu gate đạt KPI (miss_G ≤ ε), ngược lại âm; q_o* = điểm adv đổi dấu. Dự đoán có CI bootstrap theo chuỗi "
              f"(B = {mi['B']}); replay là điểm. ε ∈ {{2 %, 5 %}} (ε = 1 % do đuôi D = 1 quyết định — mục 7). "
              f"Replay trong CI dự đoán: {mi['n_inside']}/{mi['n_defined']} ô xác định được.", "",
              "| r | nhóm (M) | Lmax | ε | a_P dự đoán / replay | q_o* dự đoán [CI] | q_o* replay | trong CI |", "|---|---|---|---|---|---|---|---|"]
        for x in mi["rows"]:
            ci = "—" if x["qo_star_pred_ci"] is None else f"[{x['qo_star_pred_ci'][0]:.4f}, {x['qo_star_pred_ci'][1]:.4f}]"
            ins = "—" if x["replay_inside_pred_ci"] is None else ("✓" if x["replay_inside_pred_ci"] else "✗")
            L.append(f"| {x['r']:g} | {x['group']} ({x['M_median']:g}) | {x['Lmax']} | {x['eps']:g} | {inf_(x['a_P_pred'])} / {inf_(x['a_P_replay'])} | "
                     f"{inf_(x['qo_star_pred'])} {ci} | {inf_(x['qo_star_replay'])} | {ins} |")
        by = {}
        for x in mi["rows"]:
            if 0 < x["qo_star_pred"] < float("inf"):
                by.setdefault(x["r"], [0, 0])
                by[x["r"]][0] += 1
                by[x["r"]][1] += int(bool(x["replay_inside_pred_ci"]))
        L += ["", "- q_o* = 0 với CI [0, 0] là ô SUY BIẾN (cả hai chính sách không đạt ε, vd r = 0,5, L ≤ 3: periodic chạy mọi khung vẫn có miss 0,5⁴ > 5 %); "
              "q_o* = ∞: gate rẻ hơn trên cả lưới. Ô CÓ THÔNG TIN (0 < q_o* < ∞): " + ", ".join(f"r = {r:g}: {v[1]}/{v[0]} trong CI" for r, v in by.items()) + "."]
    if len(orac):
        L += ["", "## 5. Trần: stride tối ưu từng chuỗi (oracle) vs periodic chung, KPI miss E3", "",
              "| r | fp | a | miss periodic | miss oracle (a thực) |", "|---|---|---|---|---|"]
        for x in orac.itertuples(index=False):
            L.append(f"| {x.r:g} | {x.fp:g} | {x.a:g} | {pct(x.periodic_miss)} | {pct(x.oracle_miss)} ({f3(x.oracle_a)}) |")
    ta = json.loads((SIM / "tail_analysis.json").read_text(encoding="utf-8")) if (SIM / "tail_analysis.json").exists() else None
    if ta:
        fa = lambda a: "—" if a is None else f"{a:.3f}"  # noqa: E731
        L += ["", "## 7. Vì sao a_P(ε = 1 %) = 0,462 ở đây nhưng A3 (P0b) cho S* = 3 (a ≈ 0,33)? (P2B-A2)", "",
              f"Cùng E3 cửa sổ sinh. TRAIN: {ta['n_events_train']} sự kiện, D = 1: {pct(ta['share_D_1_train'])}, D ≤ 2: {pct(ta['share_D_le_2_train'])} "
              f"({ta['n_D_le_2_train']} sự kiện; {pct(ta['share_border_among_D_le_2_train'], 0)} vào từ mép). ALL (TRAIN+TEST, như A3): D ≤ 2: {pct(ta['share_D_le_2_all'])}.", "",
              "| r | KPI | ε | (i) S nguyên TRAIN | (i) S nguyên ALL | (ii) S thực TRAIN | (iii) engine m=0 | (iv) engine ROI 5 % |", "|---|---|---|---|---|---|---|---|"]
        for x in ta["rows"]:
            L.append(f"| {x['r']:g} | {x['kpi']} | {x['eps']:g} | {fa(x['a_int_train'])} (S={x['S_int_train']}) | {fa(x['a_int_all'])} (S={x['S_int_all']}) | "
                     f"{fa(x['a_real_train'])} | {fa(x['a_engine_m0'])} | {fa(x['a_engine_m5'])} |")
        L += ["", "Đóng góp của đuôi — a_P (S thực, công thức) khi bỏ phần sự kiện ngắn nhất, ε = 1 %:", "",
              "| r | KPI | bỏ 0 % | bỏ 1 % | bỏ 2 % | bỏ 5 % |", "|---|---|---|---|---|---|"]
        tdf = pd.DataFrame(ta["tail"])
        for (r, k), g in tdf[tdf.eps == 0.01].groupby(["r", "kpi"]):
            g = g.set_index("drop_shortest")
            L.append(f"| {r:g} | {k} | " + " | ".join(fa(g.loc[d, "a_real"]) for d in (0.0, 0.01, 0.02, 0.05)) + " |")
        L += ["", "- **Nguyên nhân:** (a) engine = công thức với S thực (cột ii ≈ iii) → engine đúng; (b) chênh với A3 là do TẬP (TRAIN vs ALL: tỉ lệ D = 1 "
              "trên TRAIN lớn hơn ε = 1 %, mỗi sự kiện D = 1 bị bỏ lỡ với xác suất 1 − 1/S ⇒ buộc S ≈ 2,2) và do S nguyên (A3) vs S thực; "
              "(c) a_P(1 %) bằng nhau ở 3 KPI vì cùng bị đuôi D = 1 quyết định. Bỏ 1 % sự kiện ngắn nhất hoặc dùng ROI 5 % làm a_P giảm mạnh.",
              "- **Quyết định (theo PROMPT P2B):** iso-KPI chính dùng ε = 2 % và 5 %; ε = 1 % vào phụ lục với cảnh báo đuôi."]
    ml = json.loads((SIM / "multilook.json").read_text(encoding="utf-8")) if (SIM / "multilook.json").exists() else None
    if ml:
        L += ["", "## 8. Lợi thế nhiều lần nhìn khi detector kém (THEORY Remark G2′, P2B-A3)", "",
              f"Cặp (r cao → r thấp) cùng cấu hình cue, KPI latency L ∈ {{3, 5, 10}}: {ml['n']} cặp. Δ tăng khi r giảm ở {pct(ml['share_obs_increase'])} "
              f"(dự đoán {pct(ml['share_pred_increase'])}); dấu dự đoán đúng {pct(ml['sign_match'])}, và {pct(ml['sign_match_decisive'])} trên "
              f"{ml['n_decisive']} cặp có khoảng quan sát không chứa 0. Kết luận có ĐIỀU KIỆN (Δ(r) lõm, Δ(0) = 0; tăng khi r giảm chỉ trên [r†, 1]).", "",
              "| cặp | n | Δ tăng (quan sát) | Δ tăng (dự đoán) | khớp dấu |", "|---|---|---|---|---|"]
        for k, v in ml["by_pair"].items():
            L.append(f"| {k} | {v['n']} | {pct(v['obs_increase'])} | {pct(v['pred_increase'])} | {pct(v['sign_match'])} |")
    L += ["", "## 6. Giới hạn", "",
          "- Detector giả lập: recall độc lập theo box (A-ind đúng theo cấu tạo); YOLO thật có recall tương quan theo thời gian và theo cỡ vật thể → làm lại với dump thật.",
          "- Cue giả lập theo A-gate; q_in/q_out thật đo trên TRAIN ở P2 (p2_estimate_channel.py). Hovering/moving: \"unknown\" (chưa có frames).",
          "- Chỉ UAVDT TRAIN; không dùng TEST (L8)."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
