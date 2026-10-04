"""P4 figures for the manuscript -> paper/figures/*.pdf (style of code/p2/p2_figures.py: STIX 8 pt, 3.5 in / 7.16 in).

Data are read from P3 / P3b outputs (not recomputed): results/p3/cells_test.csv, results/p3/exploratory/e1,e2,e4 json,
results/p2/channel_train.json. Captions in the paper state split and pre-registered / post hoc.
  fig_p3_boundary_map   confirmatory cells x prediction x result (TEST, pre-registered)          3.5 in
  fig_p3_forest         Delta-hat (H3/H5) and DeltaDelta-hat (H8) with CI vs prediction (TEST)   7.16 in
  fig_p3_iso_qostar     iso-KPI boundary q_o*(M), Cor. G5 with the frozen TRAIN channel           3.5 in
  fig_p3b_calibration   predicted vs observed (post hoc)                                          3.5 in
  fig_p3b_recall_vs_k   recall after onset r(k) vs constant model r (post hoc)                    3.5 in
  fig_p3b_iso_feasible  latency-miss floors and relaxed iso-KPI (post hoc)                        7.16 in
Run: python code/p4/p4_figures.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "p2"))
import p2_figures as F  # noqa: E402

R3 = ROOT / "results" / "p3"
EX = R3 / "exploratory"
DBL_W = 7.16
VC = {"correct": F.BLUES[1], "wrong": "#b2182b", "inconclusive": F.LIGHT}
CUE = {"border_band": "border band", "orb_lite_diff": "ORB-lite diff.", "raw_diff": "raw diff.", "tiny_det": "tiny det."}
CUE_C = {"border_band": F.BLUES[0], "orb_lite_diff": F.BLUES[2], "raw_diff": F.GREY, "tiny_det": "#b2182b"}


def save(fig, name, width):
    out = F.save(fig, name) if width <= F.COL_W + 0.01 else _save_wide(fig, name)
    return out


def _save_wide(fig, name):
    F.FIG.mkdir(parents=True, exist_ok=True)
    F.PREV.mkdir(parents=True, exist_ok=True)
    fig.savefig(F.FIG / f"{name}.pdf")
    fig.savefig(F.PREV / f"{name}.png", dpi=110)
    plt.close(fig)
    print("wrote", f"paper/figures/{name}.pdf (double column)")


def short(x):
    cue = {"border_band": "border", "orb_lite_diff": "ORB-lite", "raw_diff": "raw", "tiny_det": "tiny"}[x["cue"]]
    eg = "hov." if x["ego"] == "hovering" else "mov."
    mb = "$M$ 6–20" if x["M_bin"] == "6-20" else "$M>20$"
    return f"{cue}, {eg} {mb}"


def fig_map(d):
    d = d.copy()
    d["row"] = [short(x) for x in d.to_dict("records")]
    d["col"] = [{"H2": "H2", "H3-H5": "H3/H5", "H8": "H8"}[g] + " " + ("$L$=3" if str(k).endswith("3") else "$L$=5") for g, k in zip(d.group, d.kpi)]
    order = d.sort_values(["cue", "ego", "M_bin"]).row.unique().tolist()
    cols = [f"{g} $L$={L}" for g in ("H2", "H3/H5", "H8") for L in (3, 5)]
    fig, ax = plt.subplots(figsize=(2.9, 3.3))
    for x in d.to_dict("records"):
        i, j = order.index(x["row"]), cols.index(x["col"])
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=VC[x["verdict"]], ec="white", lw=0.8))
        ax.text(j + 0.5, i + 0.52, "+" if x["sign_pred"] == "+" else "≤0", ha="center", va="center", fontsize=7,
                color="white" if x["verdict"] != "inconclusive" else "black")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(order), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)) + 0.5, order)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=VC[v]) for v in VC]
    ax.legend(hs, ["correct", "wrong", "inconclusive"], ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0), handlelength=1, columnspacing=1)
    return save(fig, "fig_p3_boundary_map", F.COL_W)


def fig_forest(d):
    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.7))  # P4d: height -25 % (was 3.6 in)
    for ax, g, lab in ((axs[0], "H3-H5", r"$\hat\Delta=\mathrm{miss}_P-\mathrm{miss}_G$ (matched cost)"),
                       (axs[1], "H8", r"$\widehat{\Delta\Delta}=\hat\Delta(\mathrm{n@640})-\hat\Delta(\mathrm{s@1024})$")):
        x = d[d.group == g].sort_values(["kpi", "cue", "ego", "M_bin"]).reset_index(drop=True)
        for i, r in enumerate(x.to_dict("records")):
            c = VC[r["verdict"]] if r["verdict"] != "inconclusive" else F.GREY
            ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=1.0)
            ax.plot(r["obs_value"], i, "o", color=c, ms=3)
            ax.plot(r["pred_value"], i, "x", color="black", ms=3.5, mew=0.8)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(0.005, color="black", lw=0.4, ls=":")
        ax.set_yticks(range(len(x)), [("$L$=3 " if str(k).endswith("3") else "$L$=5 ") + short(r) for k, r in zip(x.kpi, x.to_dict("records"))],
                      fontsize=5.5)
        ax.set_ylim(len(x) - 0.5, -0.5)
        ax.set_xlabel(lab)
    axs[0].plot([], [], "x", color="black", label="TRAIN prediction")
    for v in ("correct", "wrong"):
        axs[0].plot([], [], "o-", color=VC[v], label=v)
    axs[0].plot([], [], "o-", color=F.GREY, label="inconclusive")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return save(fig, "fig_p3_forest", DBL_W)


def fig_iso(d):
    ch = json.loads((ROOT / "results" / "p2" / "channel_train.json").read_text(encoding="utf-8"))
    rho1 = ch["rho_onset"]["1"]
    ops = {o["cue"]: o for o in ch["operating_points"] if o.get("theta_star")}
    Ms = np.arange(1, 61)
    eps = 0.02
    fig, axs = plt.subplots(1, 2, figsize=(F.COL_W, 2.0), sharey=True)
    for ax, L in zip(axs, (3, 5)):
        T = L + 1
        for cue, op in sorted(ops.items()):
            for eg, ls in (("hovering", "-"), ("moving", "--")):
                qb = op["q_b"][eg]
                qs = ((1 - eps) / T - rho1 - op["c"]) / (1 - rho1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    qo = 1 - ((1 - qs) / (1 - qb)) ** (1 / Ms) if qs > qb else np.full(len(Ms), np.nan)
                ax.plot(Ms, qo, ls=ls, color=CUE_C[cue], lw=0.9, label=f"{CUE[cue]}" if (L == 3 and eg == "hovering") else None)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 0.3)
        ax.set_title(f"$L_{{max}}$={L}, $\\varepsilon$=2%")
        ax.set_xlabel("$M$")
    axs[0].set_ylabel(r"$q_o^*$ (gate cheaper below)")
    axs[0].legend(fontsize=6, loc="upper right", handlelength=1.4)
    fig.tight_layout()
    return save(fig, "fig_p3_iso_qostar", F.COL_W)


def fig_calibration():
    e4 = json.loads((EX / "e4_calibration.json").read_text(encoding="utf-8"))
    d = pd.DataFrame(e4["rows"])
    fig, axs = plt.subplots(1, 2, figsize=(F.COL_W, 1.9))
    for ax, g, key, lab in ((axs[0], "H3-H5", "H3H5", r"\Delta"), (axs[1], "H8", "H8", r"\Delta\Delta")):
        x = d[d.group == g]
        for v in ("correct", "wrong", "inconclusive"):
            s = x[x.verdict == v]
            c = VC[v] if v != "inconclusive" else F.GREY
            ax.errorbar(s.pred_value, s.obs_value, yerr=[s.obs_value - s.lo, s.hi - s.obs_value], fmt="o", ms=2.5, color=c, lw=0.5,
                        elinewidth=0.4, capsize=0)
        lim = [min(x.pred_value.min(), x.lo.min()), max(x.pred_value.max(), x.hi.max())]
        ax.plot(lim, lim, ":", color="black", lw=0.6)
        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.4)
        ax.set_xlabel(f"${lab}_{{pred}}$ (TRAIN)")
        ax.set_ylabel(f"$\\hat{{{lab}}}$ (TEST)" if g == "H3-H5" else "")
        ax.set_title(f"{g.replace('-', '/')}: Spearman {e4[key]['spearman']:.2f}")
    fig.tight_layout()
    return save(fig, "fig_p3b_calibration", F.COL_W)


def fig_recall():
    e2 = json.loads((EX / "e2_recall_k.json").read_text(encoding="utf-8"))
    d = pd.DataFrame(e2["rows"])
    fig, axs = plt.subplots(1, 2, figsize=(F.COL_W, 2.0), sharey=True)
    for ax, lv, ttl in zip(axs, ("yolo26s_1024", "yolo26n_640"), ("yolo26s@1024", "yolo26n@640")):
        for (split, typ), g in d[(d.level == lv) & (d.e3_type != "all")].groupby(["split", "e3_type"]):
            g = g.sort_values("k")
            c = F.BLUES[1] if typ == "border" else "#b2182b"
            ax.plot(g.k, g.r, "-" if split == "test" else "--", color=c, lw=0.9, label=f"{split.upper()} {typ}")
            if split == "test":
                ax.fill_between(g.k, g.lo, g.hi, color=c, alpha=0.12, lw=0)
        ax.axhline(e2["r_model"][lv], color="black", lw=0.6, ls=":")
        ax.text(10, e2["r_model"][lv] + 0.01, "model $r$", ha="right", va="bottom", fontsize=6.5)
        ax.set_title(ttl)
        ax.set_xlabel("frames since onset $k$")
        ax.set_ylim(0.3, 1.0)
    axs[0].set_ylabel("recall $r(k)$")
    axs[0].legend(fontsize=6, loc="lower left", handlelength=1.6)
    fig.tight_layout()
    return save(fig, "fig_p3b_recall_vs_k", F.COL_W)


def fig_feasible():
    e1 = json.loads((EX / "e1_feasibility.json").read_text(encoding="utf-8"))
    s = pd.DataFrame(e1["strata"])
    it = pd.DataFrame(e1["iso"])
    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]
    lab = [("hov." if r.ego == "hovering" else "mov.") + "\n" + ("6–20" if r.M_bin == "6-20" else ">20") + f"\n$L$={r.L}" for r in s.itertuples()]
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.miss_P_floor_S1, 0.4, color=F.BLUES[0], label="periodic, $S=1$")
    ax.bar(x + 0.2, s.gate_floor, 0.4, color=F.BLUES[2], label="best gate on TRAIN $\\theta$ grid")
    ax.axhline(0.02, color="#b2182b", lw=0.8, ls="--", label="frozen $\\varepsilon$ = 2%")
    ax.set_xticks(x, lab, fontsize=6.5)
    ax.set_ylabel("latency-miss floor")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    ax = axs[1]
    keys = list(it.groupby(["ego", "M_bin", "L"]).groups)
    for i, key in enumerate(keys):
        g = it[(it.ego == key[0]) & (it.M_bin == key[1]) & (it.L == key[2])]
        for j, cue in enumerate(sorted(CUE)):
            for k, r in enumerate(g[g.cue == cue].sort_values("eps").itertuples()):
                y = i + 0.18 * j - 0.27 + 0.04 * k
                if r.diff_lo is not None and r.diff_hi is not None and np.isfinite(r.diff):
                    ax.plot([r.diff_lo, r.diff_hi], [y, y], color=CUE_C[cue], lw=0.6)
                    ax.plot(r.diff, y, "o", color=CUE_C[cue], ms=1.5 + 0.8 * k)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(keys)), [f"{'hov.' if e == 'hovering' else 'mov.'} {'6–20' if m == '6-20' else '>20'}, $L$={L}" for e, m, L in keys])
    ax.set_xlabel(r"$a_G-a_P(\varepsilon)$, $\varepsilon\in\{$floor+2 pt, 10, 15, 20%$\}$ (marker size)")
    for cue in sorted(CUE):
        ax.plot([], [], "o-", color=CUE_C[cue], label=CUE[cue], ms=2)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return save(fig, "fig_p3b_iso_feasible", DBL_W)


def main():
    F.style()
    d = pd.read_csv(R3 / "cells_test.csv")
    fig_map(d)
    fig_forest(d)
    fig_iso(d)
    fig_calibration()
    fig_recall()
    fig_feasible()


if __name__ == "__main__":
    main()
