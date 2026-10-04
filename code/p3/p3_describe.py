"""P3-C: phần MÔ TẢ sau khi chấm họ xác nhận (không ảnh hưởng B). Không có ô nào ở đây được tính vào tiêu chí.

C1 Ô ngoài họ (prereg_cells.json, có dự đoán đóng băng): ε = 5 % / 1 %, L = 10, L = 30, KPI miss, R = ∞, M 1–5, moving > 20, H8 gộp
   → cùng luật 3 mức (p2_prereg_score.judge / judge_iso) chỉ để mô tả; ô < 30 sự kiện hoặc < 3 chuỗi TEST ghi "dưới cỡ mẫu".
C2 Độ nhạy (biến thể replay, chỉ ô họ xác nhận): onset_alt, ROI m = 5 %, score 0,05 / 0,50 → Δ̂ và nhãn 3 mức theo biến thể.
C3 Biểu đồ (paper/figs/): fig_p3_boundary_map (ô × dự đoán × kết quả), fig_p3_forest (Δ̂ ± CI theo ô), fig_p3_iso_qostar
   (đường iso-KPI q_o*(M) theo Cor. G5 với kênh TRAIN đóng băng + quan sát H2 TEST).
Xuất: results/p3/desc_test.csv, results/p3/sensitivity_test.csv, results/p3/describe_test.json, paper/figs/*.pdf|png.
Chạy: python code/p3/p3_describe.py
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_prereg_score import judge  # noqa: E402

OUT = ROOT / "results" / "p3"
FIGS = ROOT / "paper" / "figs"
CELLS = ROOT / "results" / "p2" / "prereg_cells.json"
VCOL = {"correct": "#2e7d32", "wrong": "#c62828", "inconclusive": "#9e9e9e", "under_size": "#ffffff"}
VI = {"correct": "ĐÚNG", "wrong": "SAI", "inconclusive": "CHƯA KẾT LUẬN", "under_size": "dưới cỡ mẫu"}


def verdict_row(x):
    if not x["size_rule_test"]:
        return "under_size"
    if x["comparison"] == "iso_kpi":
        s_obs = "+" if x["gate_cheaper"] else "<=0"
        return "correct" if s_obs == x["sign_pred"] else "wrong"   # mô tả: không có CI (ngoài họ) → chỉ dấu
    if x["sign_pred"] == "—" or pd.isna(x.get("lo")):
        return None
    return judge(x["sign_pred"], x["lo"], x["hi"])


def desc_kind(x):
    k = []
    if x["R"] == "inf":
        k.append("R=inf")
    if x["comparison"] == "iso_kpi" and x.get("eps") is not None and not pd.isna(x.get("eps")) and x["eps"] != 0.02:
        k.append(f"eps={x['eps']:g}")
    if str(x["kpi"]).endswith("10") or x["kpi"] == "10":
        k.append("L=10")
    if str(x["kpi"]).endswith("30") or x["kpi"] == "30":
        k.append("L=30")
    if x["kpi"] == "miss":
        k.append("miss")
    if x["M_bin"] == "1-5":
        k.append("M 1-5")
    if x["ego"] == "moving" and x["M_bin"] == ">20":
        k.append("moving >20")
    if x["M_bin"] == "all":
        k.append("H8 pooled" if x["comparison"] == "r_pair" else "M all")
    if x["comparison"] != "r_pair" and x["r_level"] != "yolo26s_1024":
        k.append("r_level " + x["r_level"])
    return ", ".join(k) or "khác"


def c1():
    d = pd.read_parquet(OUT / "replay_test_all.parquet")
    d = d[~d.confirmatory].copy()
    d["verdict"] = [verdict_row(x) for x in d.to_dict("records")]
    d = d[d.verdict.notna()]
    d["kind"] = [desc_kind(x) for x in d.to_dict("records")]
    d.to_csv(OUT / "desc_test.csv", index=False, encoding="utf-8")
    summ = {}
    for key in ("R=inf", "eps=0.05", "eps=0.01", "L=10", "L=30", "miss", "M 1-5", "moving >20", "H8 pooled"):
        g = d[d.kind.str.contains(key, regex=False)]
        summ[key] = dict(n=int(len(g)), **{v: int((g.verdict == v).sum()) for v in VI})
    return d, summ


def c2():
    conf = pd.read_parquet(OUT / "replay_test_all.parquet")
    conf = conf[conf.confirmatory].set_index("id")
    rows = []
    for var in ("onset_alt", "roi5", "score005", "score050"):
        p = OUT / f"replay_test_{var}.parquet"
        if not p.exists():
            continue
        v = pd.read_parquet(p)
        v = v[v.confirmatory]
        for x in v.to_dict("records"):
            m = conf.loc[x["id"]]
            val = x.get("diff") if x["comparison"] == "iso_kpi" else x.get("delta")
            main_val = m.get("diff") if x["comparison"] == "iso_kpi" else m.get("delta")
            rows.append(dict(variant=var, id=x["id"], hypothesis=x["hypothesis"], comparison=x["comparison"], sign_pred=x["sign_pred"],
                             value=val, lo=x.get("lo"), hi=x.get("hi"), value_main=main_val, n_events=x["n_events_test"],
                             verdict=verdict_row(x)))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "sensitivity_test.csv", index=False, encoding="utf-8")
    summ = {var: {v: int((g.verdict == v).sum()) for v in VI} | {"n": int(len(g))} for var, g in s.groupby("variant")} if len(s) else {}
    return s, summ


def short(x):
    cue = {"border_band": "border", "orb_lite_diff": "orb_lite", "raw_diff": "raw", "tiny_det": "tiny"}[x["cue"]]
    eg = "hov" if x["ego"] == "hovering" else "mov"
    return f"{cue}·{eg}·M{x['M_bin']}"


def fig_map(cells):
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = cells.copy()
    cells["row"] = [short(x) for x in cells.to_dict("records")]
    cols = []
    for x in cells.to_dict("records"):
        L = str(x["kpi"]).split("_")[-1].lstrip("L")
        cols.append({"H2": "H2 iso", "H3-H5": "H3/H5", "H8": "H8"}[x["group"]] + f" L{L}")
    cells["col"] = cols
    rws = sorted(cells.row.unique())
    cls = ["H2 iso L3", "H2 iso L5", "H3/H5 L3", "H3/H5 L5", "H8 L3", "H8 L5"]
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(rws) + 1.2))
    for x in cells.to_dict("records"):
        i, j = rws.index(x["row"]), cls.index(x["col"])
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=VCOL[x["verdict"]], ec="white", lw=1))
        ax.text(j + 0.5, i + 0.5, x["sign_pred"].replace("<=0", "≤0"), ha="center", va="center", fontsize=7,
                color="white" if x["verdict"] != "inconclusive" else "black")
    ax.set_xlim(0, len(cls))
    ax.set_ylim(len(rws), 0)
    ax.set_xticks(np.arange(len(cls)) + 0.5, cls, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rws)) + 0.5, rws, fontsize=7)
    ax.tick_params(length=0)
    for k in ("top", "right", "left", "bottom"):
        ax.spines[k].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=VCOL[v]) for v in ("correct", "wrong", "inconclusive")]
    ax.legend(hs, ["correct", "wrong", "inconclusive"], fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("UAVDT TEST — confirmatory family (text = predicted sign)", fontsize=8, pad=18)
    return save(fig, "fig_p3_boundary_map")


def fig_forest(cells):
    d = cells[cells.comparison != "iso_kpi"].copy()
    d["lab"] = [f"{x['group']} L{str(x['kpi']).split('_')[-1].lstrip('L')} {short(x)}" for x in d.to_dict("records")]
    d = d.sort_values(["group", "kpi", "cue", "ego", "M_bin"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 0.17 * len(d) + 1.0))
    y = np.arange(len(d))
    for i, x in enumerate(d.to_dict("records")):
        col = VCOL[x["verdict"]]
        lo, hi = x["lo"], x["hi"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=col, lw=1.2)
        ax.plot(x["obs_value"], i, "o", color=col, ms=3)
        ax.plot(x["pred_value"], i, "x", color="black", ms=3, mew=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(0.005, color="black", lw=0.4, ls=":")
    ax.set_yticks(y, d.lab, fontsize=5.5)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlabel("Δ̂ (H3/H5) or ΔΔ̂ (H8), 95 % CI; × = TRAIN prediction; dotted = δ_min", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    return save(fig, "fig_p3_forest")


def fig_iso(cells):
    """Cor. G5 (r = 1): gate rẻ hơn ở iso-KPI iff ρ1 + (1−ρ1) q_out,eff(M) + c < (1−ε)/T; q_out,eff = 1 − (1−q_b)(1−q_o)^M.
    ⇒ q_o*(M) = 1 − ((1 − q*)/(1 − q_b))^(1/M), q* = ((1−ε)/T − ρ1 − c)/(1 − ρ1). Kênh TRAIN đóng băng (channel_train.json)."""
    ch = json.loads((ROOT / "results" / "p2" / "channel_train.json").read_text(encoding="utf-8"))
    rho1 = ch["rho_onset"]["1"]
    ops = {o["cue"]: o for o in ch["operating_points"] if o.get("theta_star")}
    Ms = np.arange(1, 61)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    eps = 0.02
    colors = dict(border_band="#1565c0", orb_lite_diff="#ef6c00", raw_diff="#6a1b9a", tiny_det="#00838f")
    for ax, L in zip(axs, (3, 5)):
        T = L + 1
        for cue, op in sorted(ops.items()):
            for eg, ls in (("hovering", "-"), ("moving", "--")):
                qb = op["q_b"].get(eg, op["q_b"]["all"])
                qs = ((1 - eps) / T - rho1 - op["c"]) / (1 - rho1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    qo = 1 - ((1 - qs) / (1 - qb)) ** (1 / Ms) if qs > 0 and (1 - qs) / (1 - qb) < 1 else np.full(len(Ms), np.nan)
                ax.plot(Ms, qo, ls=ls, color=colors[cue], lw=1, label=f"{cue} {eg}" if L == 3 else None)
            ax.plot([13, 30], [max(op["q_o"], 1e-5)] * 2, "o", color=colors[cue], ms=3)
        sub = cells[(cells.comparison == "iso_kpi") & (cells.kpi == f"iso_L{L}")]
        txt = ", ".join(f"{v}: {int((sub.verdict == v).sum())}" for v in ("correct", "wrong", "inconclusive"))
        ax.set_title(f"L = {L} (T = {T}), ε = 2 %\nTEST H2 cells: {txt}", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel("M (vehicles visible)", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("q_o* (gate cheaper below the curve)", fontsize=7)
    axs[0].legend(fontsize=5, frameon=False, loc="lower left")
    fig.suptitle("Iso-KPI boundary q_o*(M), Cor. G5, frozen TRAIN channel\n(dots: fitted q_o at M = 13, 30; floored at 1e-5)", fontsize=7)
    return save(fig, "fig_p3_iso_qostar")


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=150)
    plt.close(fig)
    return f"paper/figs/{name}.pdf"


def main():
    cells = pd.read_csv(OUT / "cells_test.csv")
    d1, s1 = c1()
    d2, s2 = c2()
    figs = [fig_map(cells), fig_forest(cells), fig_iso(cells)]
    out = dict(note="MÔ TẢ — không tính vào tiêu chí; iso ngoài họ chấm theo dấu (không CI)", desc_by_kind=s1, n_desc=int(len(d1)),
               sensitivity=s2, figures=figs)
    (OUT / "describe_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
