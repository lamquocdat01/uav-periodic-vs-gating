"""P3b E6 [POST HOC — không thuộc họ xác nhận]: RESULTS_28_EXPLORATORY.md + bảng cho bài + 3 hình (paper/figs/fig_p3b_*.pdf).

Đọc results/p3/exploratory/*.json (p3b_explore.py) và kết quả P3 (score_test.json, cells_test.csv, describe_test.json) — KHÔNG ghi đè chúng.
Chạy: python code/p3/exploratory/p3b_report.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from p3b_lib import POST_HOC  # noqa: E402

R3 = ROOT / "results" / "p3"
EX = R3 / "exploratory"
FIGS = ROOT / "paper" / "figs"
VI = {"correct": "ĐÚNG", "wrong": "SAI", "inconclusive": "CHƯA KẾT LUẬN"}


def J(name):
    return json.loads((EX / f"{name}.json").read_text(encoding="utf-8"))


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_calibration(e4):
    d = pd.DataFrame(e4["rows"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    col = {"correct": "#2e7d32", "wrong": "#c62828", "inconclusive": "#9e9e9e"}
    for ax, g, key, lab in ((axs[0], "H3-H5", "H3H5", "Δ"), (axs[1], "H8", "H8", "ΔΔ")):
        x = d[d.group == g]
        for v, c in col.items():
            s = x[x.verdict == v]
            ax.errorbar(s.pred_value, s.obs_value, yerr=[s.obs_value - s.lo, s.hi - s.obs_value], fmt="o", ms=3, color=c, lw=0.6, label=v)
        lim = [min(x.pred_value.min(), x.lo.min()), max(x.pred_value.max(), x.hi.max())]
        ax.plot(lim, lim, "k:", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        c = e4[key]
        ax.set_title(f"{g}: slope {c['slope']:.2f}, MAE {c['mae']:.3f}, Spearman {c['spearman']:.2f}", fontsize=7)
        ax.set_xlabel(f"{lab}_pred (TRAIN model)", fontsize=7)
        ax.set_ylabel(f"{lab}̂ TEST (95 % CI)", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"POST HOC calibration — {e4['all48']['n']} cells: slope {e4['all48']['slope']:.2f}, MAE {e4['all48']['mae']:.3f}, "
                 f"Spearman {e4['all48']['spearman']:.2f}", fontsize=7)
    save(fig, "fig_p3b_calibration")


def fig_recall(e2):
    d = pd.DataFrame(e2["rows"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, lv in zip(axs, ("yolo26s_1024", "yolo26n_640")):
        for (split, typ), g in d[(d.level == lv) & (d.e3_type != "all")].groupby(["split", "e3_type"]):
            g = g.sort_values("k")
            ls = "-" if split == "test" else "--"
            c = "#1565c0" if typ == "border" else "#ef6c00"
            ax.plot(g.k, g.r, ls, color=c, lw=1, marker="o", ms=2.5, label=f"{split} {typ}")
            ax.fill_between(g.k, g.lo, g.hi, color=c, alpha=0.08 if split == "train" else 0.15)
        ax.axhline(e2["r_model"][lv], color="k", lw=0.7, ls=":", label=f"model r = {e2['r_model'][lv]:.3f}")
        ax.set_title(lv, fontsize=8)
        ax.set_xlabel("k = frames since onset", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("recall r(k) (score ≥ 0.25, IoU ≥ 0.5)", fontsize=7)
    axs[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.suptitle("POST HOC — detector recall after onset vs constant r of the model", fontsize=7)
    save(fig, "fig_p3b_recall_vs_k")


def fig_iso(e1):
    s = pd.DataFrame(e1["strata"])
    it = pd.DataFrame(e1["iso"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axs[0]
    lab = [f"{r.ego[:3]} {r.M_bin} L{r.L}" for r in s.itertuples()]
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.miss_P_floor_S1, 0.4, label="periodic floor miss_P(S=1)", color="#455a64")
    ax.bar(x + 0.2, s.gate_floor, 0.4, label="gate floor min_θ miss_G", color="#90a4ae")
    ax.axhline(0.02, color="#c62828", lw=0.8, ls="--", label="ε = 2 % (frozen)")
    ax.set_xticks(x, lab, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("latency miss", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, frameon=False)
    ax = axs[1]
    cues = sorted(it.cue.unique())
    cols = dict(zip(cues, ["#1565c0", "#ef6c00", "#6a1b9a", "#00838f"]))
    for i, (key, g) in enumerate(it.groupby(["ego", "M_bin", "L"])):
        for j, cue in enumerate(cues):
            gg = g[g.cue == cue].sort_values("eps")
            for k, r in enumerate(gg.itertuples()):
                y = i + 0.15 * j - 0.25 + 0.03 * k
                if r.diff_lo is not None and r.diff_hi is not None and np.isfinite(r.diff):
                    ax.plot([r.diff_lo, r.diff_hi], [y, y], color=cols[cue], lw=0.7)
                    ax.plot(r.diff, y, "o", color=cols[cue], ms=2 + k)
    ax.axvline(0, color="k", lw=0.5)
    keys = [f"{e[:3]} {m} L{L}" for (e, m, L), _ in it.groupby(["ego", "M_bin", "L"])]
    ax.set_yticks(range(len(keys)), keys, fontsize=6)
    ax.set_xlabel("a_G − a_P(ε) (marker size: ε = floor+2pt, 10, 15, 20 %)", fontsize=6.5)
    ax.tick_params(labelsize=7)
    for cue in cues:
        ax.plot([], [], color=cols[cue], label=cue)
    ax.legend(fontsize=5.5, frameon=False)
    fig.suptitle("POST HOC — iso-KPI feasibility on TEST (yolo26s@1024)", fontsize=7)
    save(fig, "fig_p3b_iso_feasible")


def main():
    e1, e2, e3, e4, e5 = (J(n) for n in ("e1_feasibility", "e2_recall_k", "e3_qostar", "e4_calibration", "e5_drift"))
    sc = json.loads((R3 / "score_test.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(R3 / "cells_test.csv")
    desc = json.loads((R3 / "describe_test.json").read_text(encoding="utf-8"))
    fig_calibration(e4)
    fig_recall(e2)
    fig_iso(e1)
    L = ["# RESULTS_28_EXPLORATORY — phân tích hậu kiểm sau khi mở TEST (sinh tự động)", "",
         f"*Sinh bởi `code/p3/exploratory/p3b_report.py`. Mọi mục dưới đây là {POST_HOC}: không đổi nhãn, tiêu chí hay kết luận của "
         "RESULTS_28.md (họ xác nhận đóng băng). Số liệu: results/p3/exploratory/*.json.*", ""]
    # E1
    L += [f"## {POST_HOC} E1 — độ khả thi iso-KPI (vì sao 24 ô H2 đúng một cách tầm thường)", "",
          "| tầng | L | n sự kiện | sàn periodic miss_P(S = 1) = ε_feasible | sàn gate min_θ miss_G (cue, θ) | a_P(2 %) dự đoán PREREG (theo cue) | a_P(2 %) quan sát |",
          "|---|---|---|---|---|---|---|"]
    for r in e1["strata"]:
        best = min(r["gate_floor_by_cue"].items(), key=lambda kv: kv[1]["miss_G"])
        pr = ", ".join(f"{c} {f(v)}" for c, v in sorted(r["pred_a_P_iso_2pct"].items()))
        L.append(f"| {r['ego']} {r['M_bin']} | {r['L']} | {r['n_events']} | {pct(r['miss_P_floor_S1'])} | {pct(r['gate_floor'])} ({best[0]}, θ {best[1]['theta']:.4g}) | {pr} | ∞ |")
    L += ["", "Iso-KPI ở ε nới (θ* đóng băng, R = 30; CI bootstrap theo chuỗi B = 1000 seed 42; **không chấm điểm**):", "",
          "| tầng | L | ε | cue | miss_G | gate đạt ε | a_G | a_P(ε) | a_G − a_P | CI 95 % | dấu |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in e1["iso"]:
        L.append(f"| {r['ego']} {r['M_bin']} | {r['L']} | {pct(r['eps'])} ({r['eps_kind']}) | {r['cue']} | {f(r['miss_G'])} | "
                 f"{'có' if r['gate_meets'] else 'không'} ({pct(r['share_gate_meets_boot'])} mẫu boot) | {f(r['a_G'])} | {f(r['a_P_iso'])} | "
                 f"{f(r['diff'], sign=True)} | [{f(r['diff_lo'], sign=True)}; {f(r['diff_hi'], sign=True)}] | {r['sign']} |")
    a, b = e1["share_correct_all"], e1["share_correct_excl_H2"]
    L += ["", f"- Tỉ lệ ĐÚNG / KẾT LUẬN ĐƯỢC của họ xác nhận: **{a['correct']}/{a['decided']} = {pct(a['share'])}** (P3); "
          f"bỏ 24 ô H2: **({a['correct']} − 24)/({a['decided']} − 24) = {b['correct']}/{b['decided']} = {pct(b['share'])}**.", ""]
    # E2
    d2 = pd.DataFrame(e2["rows"])
    L += [f"## {POST_HOC} E2 — recall theo khung kể từ onset r(k)", "",
          f"Mô hình dùng r hằng: yolo26s_1024 = {e2['r_model']['yolo26s_1024']:.3f}, yolo26n_640 = {e2['r_model']['yolo26n_640']:.3f} (mọi box VISIBLE, TRAIN).", "",
          "| split | mức | loại | " + " | ".join(f"k={k}" for k in range(11)) + " |", "|---|---|---|" + "---|" * 11]
    for (sp, lv, typ), g in d2.groupby(["split", "level", "e3_type"]):
        g = g.set_index("k")
        L.append(f"| {sp} | {lv} | {typ} | " + " | ".join(f"{g.r.get(k, float('nan')):.3f}" for k in range(11)) + " |")
    sz = pd.DataFrame(e2["size"])
    L += ["", "Kích thước box GT trung vị √(w·h) (px) theo k:", "", "| split | loại | " + " | ".join(f"k={k}" for k in range(11)) + " |", "|---|---|" + "---|" * 11]
    for (sp, typ), g in sz.groupby(["split", "e3_type"]):
        g = g.set_index("k")
        L.append(f"| {sp} | {typ} | " + " | ".join(f(g.sqrt_area_median.get(k), 1) for k in range(11)) + " |")
    # E3
    L += ["", f"## {POST_HOC} E3 — tiêu chí 2 (mô tả): q_o* quan sát vs Thm P3-iii", "",
          f"**{e3['note']}.** Cue mật độ tổng hợp (P2B-A1) trên detector thật TEST; điểm dự đoán = Thm P3-iii (r = 1, q_b = 0, c = 0, ρ1 TRAIN = {e3['rho1_train']:.4f}).", "",
          "| nhóm M | M trung vị | n sự kiện | L | ε | sàn periodic | a_P(ε) | q_o* quan sát | CI 95 % | q_o* dự đoán | trong CI |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in e3["groups"]:
        ci = r["qo_star_obs_ci"]
        ins = "—" if ci is None else ("có" if ci[0] <= r["qo_star_pred"] <= ci[1] else "không")
        L.append(f"| {r['group']} | {r['M_median']:.0f} | {r['n_events']} | {r['L']} | {pct(r['eps'])} ({r['eps_kind']}) | {pct(r['miss_P_floor'])} | {f(r['a_P_iso'])} | "
                 f"{f(r['qo_star_obs'], 4)} | {'—' if ci is None else f'[{ci[0]:.4f}; {ci[1]:.4f}]'} | {r['qo_star_pred']:.4f} | {ins} |")
    # E4
    L += ["", f"## {POST_HOC} E4 — hiệu chuẩn mô hình (Δ_pred TRAIN vs Δ̂ TEST, 48 ô H3/H5 + H8)", "",
          "| tập | n | slope | intercept | MAE | lệch TB (quan sát − dự đoán) | Spearman |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("all48", "H3/H5 + H8"), ("H3H5", "H3/H5 (Δ)"), ("H8", "H8 (ΔΔ)")):
        c = e4[k]
        L.append(f"| {lab} | {c['n']} | {c['slope']:.3f} | {c['intercept']:+.3f} | {c['mae']:.3f} | {c['bias']:+.3f} | {c['spearman']:.3f} (p {c['spearman_p']:.2g}) |")
    L += ["", "| nhóm | dự đoán | dấu quan sát (điểm, ngưỡng δ_min) | n |", "|---|---|---|---|"]
    for k, n in e4["sign_table"].items():
        g, p, o = k.split("|")
        L.append(f"| {g} | {p.replace('pred ', '')} | {o.replace('obs ', '')} | {n} |")
    u, ex = e4["H3H5_underestimate_loss"], e4["example_L5_border_hov_6_20"]
    L += ["", f"- H3/H5: Δ̂ < Δ_pred ở {u['n_obs_below_pred']}/{u['n']} ô; trung vị (Δ̂ − Δ_pred) = {u['median_obs_minus_pred']:+.3f} "
          f"(mô hình đánh giá thấp mức thua của gate). Ví dụ L5 border_band × hovering × 6–20: Δ_pred {ex['pred']:+.3f} vs Δ̂ {ex['obs']:+.3f} "
          f"[{ex['lo']:+.3f}; {ex['hi']:+.3f}].", ""]
    # E5
    L += [f"## {POST_HOC} E5 — trôi kênh TRAIN → TEST tại θ* đóng băng", "",
          "| split | λ /khung | ρ1 | ρ3 | ρ5 | ρ10 | E3 (mọi onset) |", "|---|---|---|---|---|---|---|"]
    for sp, s in e5["splits"].items():
        ro = s["rho_onset"]
        L.append(f"| {sp} | {s['lambda_per_frame']:.4f} | {ro['1']:.4f} | {ro['3']:.4f} | {ro['5']:.4f} | {ro['10']:.4f} | {s['n_onsets']} |")
    cr = pd.DataFrame(e5["cue_rows"])
    L += ["", "w = 5 (θ*):", "", "| cue | ego | q_in TRAIN | q_in TEST | q_out TRAIN | q_out TEST | J TRAIN | J TEST | a_G dự đoán TRAIN | a_G dự đoán TEST |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    w5 = cr[cr.w == 5].set_index(["cue", "ego", "split"])
    for cue in sorted(cr.cue.unique()):
        for eg in ("all", "hovering", "moving"):
            a_, b_ = w5.loc[(cue, eg, "train")], w5.loc[(cue, eg, "test")]
            L.append(f"| {cue} | {eg} | {a_.q_in:.4f} | {b_.q_in:.4f} | {a_.q_out:.4f} | {b_.q_out:.4f} | {a_.J:+.4f} | {b_.J:+.4f} | {a_.a_G_pred:.4f} | {b_.a_G_pred:.4f} |")
    sr = pd.DataFrame(e5["strata_rows"])
    L += ["", "q_in theo tầng (cửa sổ w = 5 của sự kiện E3 cửa sổ sinh thuộc tầng), q_out của ego, P(cue bắn ≥ 1 lần ở k = 0…3):", "",
          "| cue | tầng | split | n sự kiện | q_in tầng | q_out ego | J tầng | P(bắn, k ≤ 3) |", "|---|---|---|---|---|---|---|---|"]
    for x in sr.sort_values(["cue", "ego", "M_bin", "split"]).itertuples():
        L.append(f"| {x.cue} | {x.ego} {x.M_bin} | {x.split} | {x.n_events} | {x.q_in_stratum:.4f} | {x.q_out_ego:.4f} | {x.q_in_stratum - x.q_out_ego:+.4f} | {f(x.p_fire_any_k0_3)} |")
    L += ["", "Ô border_band (matched cost, R = 30): dự đoán TRAIN vs quan sát TEST:", "",
          "| tầng | L | miss_P dự đoán | miss_P TEST | miss_G dự đoán | miss_G TEST | Δ_pred | Δ̂ [CI] | a_G dự đoán | a_G TEST |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in e5["border_band_cells"]:
        L.append(f"| {x['ego']} {x['M_bin']} | {x['kpi']} | {x['miss_P_pred']:.3f} | {x['miss_P_obs']:.3f} | {x['miss_G_pred']:.3f} | {x['miss_G_obs']:.3f} | "
                 f"{x['delta_pred']:+.3f} | {x['delta_obs']:+.3f} [{x['lo']:+.3f}; {x['hi']:+.3f}] | {x['a_G_pred']:.3f} | {x['a_G_obs']:.3f} |")
    # E6
    L += ["", f"## {POST_HOC} E6 — bảng cho bài (lấy từ P3, không tính lại)", "", "(a) Họ xác nhận, 3 mức theo nhóm:", "",
          "| nhóm | n | ĐÚNG | SAI | CHƯA KẾT LUẬN | ĐÚNG / KẾT LUẬN ĐƯỢC |", "|---|---|---|---|---|---|"]
    for g, d in sc["by_group_counts"].items():
        L.append(f"| {g} | {d['n']} | {d['correct']} | {d['wrong']} | {d['inconclusive']} | {pct(d['correct'] / max(d['correct'] + d['wrong'], 1))} |")
    tc = {v: sum(d[v] for d in sc["by_group_counts"].values()) for v in VI}
    L.append(f"| tổng | 72 | {tc['correct']} | {tc['wrong']} | {tc['inconclusive']} | {pct(a['share'])} |")
    L.append(f"| tổng, loại H2 | 48 | {tc['correct'] - 24} | {tc['wrong']} | {tc['inconclusive']} | {pct(b['share'])} |")
    L += ["", "(b) 21 ô dự đoán \"+\":", "", "| nhóm | KPI | ego | M_bin | cue | giá trị dự đoán | quan sát | CI 95 % | kết quả |", "|---|---|---|---|---|---|---|---|---|"]
    for x in cells[cells.sign_pred == "+"].itertuples():
        L.append(f"| {x.group} | {x.kpi} | {x.ego} | {x.M_bin} | {x.cue} | {x.pred_value:+.4f} | {x.obs_value:+.4f} | [{x.lo:+.4f}; {x.hi:+.4f}] | {VI[x.verdict]} |")
    L += ["", "(c) Biến thể mô tả trên 72 ô (P3 describe_test.json; H2 chỉ theo dấu):", "", "| biến thể | ĐÚNG | SAI | CHƯA KẾT LUẬN |", "|---|---|---|---|"]
    for var, g in desc["sensitivity"].items():
        L.append(f"| {var} | {g['correct']} | {g['wrong']} | {g['inconclusive']} |")
    L += ["", "Hình: paper/figs/fig_p3b_calibration.pdf, fig_p3b_recall_vs_k.pdf, fig_p3b_iso_feasible.pdf."]
    (ROOT / "RESULTS_28_EXPLORATORY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote RESULTS_28_EXPLORATORY.md", len(L), "dòng")


if __name__ == "__main__":
    main()
