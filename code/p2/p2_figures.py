"""Figures for the T-ITS paper skeleton -> paper/figures/*.pdf (+ small PNG previews in paper/figures/preview/).

F1 fig_exposure.pdf   share_short (= miss at r=1) and miss_r vs stride S, E2 / E3, birth window, CI  (results/p0/exposure_grid.csv)
F2 fig_isomiss.pdf    iso-miss S*(eps) and equivalent detector rate F/S*, r in {1, 0.8, 0.5}, m=0   (results/p0/isomiss.csv)
F3 fig_onset.pdf      rho_onset(w) and ideal-gate activation bound vs periodic a_P(eps)=(1-eps)/T    (results/p0_stats.json p0b.rho_onset)
F4 fig_mstar.pdf      M* map (Lmax x q_o; R=inf, q_b=0) and M_iso map (eps=1%, c=0, q_b=0)          (results/p1/theory_checks.json)
F5 fig_sim_map.pdf    gate win/lose map with the simulated detector                                  (results/p2/sim/agnostic_cells.csv; skipped if absent)

Style: single-column width 3.5 in, 8 pt fonts (STIX serif, TrueType-embedded), one colour family (Blues) + greys.
Run: .venv/Scripts/python.exe code/p2/p2_figures.py
"""
from __future__ import annotations

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
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
PREV = FIG / "preview"
SIM_DIR = RES / "p2" / "sim"

COL_W = 3.5  # in, IEEE single column
BLUES = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
GREY = "#6f6f6f"
LIGHT = "#d9d9d9"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "lines.markersize": 3.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "errorbar.capsize": 1.5,
    })


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    PREV.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(PREV / f"{name}.png", dpi=110)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    wid = bb.width + 2 * plt.rcParams["savefig.pad_inches"]
    plt.close(fig)
    flag = "" if wid <= COL_W + 0.01 else "  WARNING: wider than one column -> fonts shrink below 8 pt"
    print(f"wrote {out.relative_to(ROOT)}  ({wid:.2f} x {bb.height:.2f} in){flag}")
    return out


def hz_ticks(sec_axis, axis: str) -> None:
    """Explicit, plainly formatted detector-rate ticks on a secondary (inverted, log) F/S axis."""
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter, NullLocator
    ax = sec_axis.yaxis if axis == "y" else sec_axis.xaxis
    ax.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_minor_locator(NullLocator())
    ax.set_minor_formatter(NullFormatter())


def fps_uavdt() -> int:
    d = json.loads((RES / "p0_stats.json").read_text(encoding="utf-8"))
    return int(d["constants"]["expected"]["UAVDT"]["fps"])


# ----------------------------------------------------------------------------- F1
def fig_exposure() -> Path:
    df = pd.read_csv(RES / "p0" / "exposure_grid.csv")
    df = df[(df.dataset == "UAVDT") & (df.vis_def == "main") & (df.group == "VEHICLE") & (df.window == "birth_W120")]
    W = json.loads((RES / "p0_stats.json").read_text(encoding="utf-8"))["meta"]["W_birth"]
    df = df[df.S <= W]  # birth-window estimate is exact for r=1 only when S <= W (THEORY_28 Cor. 1)
    F = fps_uavdt()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2), sharey=True)
    series = [("miss_r1.0", r"$r=1$ (share$_{short}$)", BLUES[0], "o"),
              ("miss_r0.8", r"$r=0.8$", BLUES[1], "s"),
              ("miss_r0.5", r"$r=0.5$", BLUES[2], "^")]
    for ax, ev, tag in zip(axes, ["E2", "E3"], ["(a)", "(b)"]):
        g = df[df.event == ev].sort_values("S")
        full = g.groupby("S", as_index=False).first()  # exposure depends on S = kR only
        ci = g.dropna(subset=["share_short_lo"]).groupby("S", as_index=False).first()
        for col, lab, c, mk in series:
            ax.plot(full.S, 100 * full[col], color=c, lw=0.9, label=lab, zorder=2)
            yerr = 100 * np.vstack([ci[col] - ci[col + "_lo"], ci[col + "_hi"] - ci[col]])
            ax.errorbar(ci.S, 100 * ci[col], yerr=yerr, fmt=mk, color=c, ms=3, lw=0.7, zorder=3,
                        mfc="white" if col != "miss_r1.0" else c)
        ax.set_xscale("log")
        ax.set_xlabel(r"Stride $S$ (frames)")
        ax.set_title(f"{tag} {ev}, birth window", loc="left")
        ax.grid(True, which="major", color=LIGHT, lw=0.4)
        sec = ax.secondary_xaxis("top", functions=(lambda s: F / np.maximum(s, 1e-9), lambda h: F / np.maximum(h, 1e-9)))
        sec.set_xlabel(f"Detector rate at {F} fps (Hz)", fontsize=8)
        hz_ticks(sec, "x")
    axes[0].set_ylabel("Events with no successful look (%)")
    axes[0].legend(loc="upper left", handlelength=1.4)
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig_exposure")


# ----------------------------------------------------------------------------- F2
def fig_isomiss() -> Path:
    df = pd.read_csv(RES / "p0" / "isomiss.csv")
    df = df[(df.dataset == "UAVDT") & (df.m == 0.0) & (df.estimator == "birth")]
    F = fps_uavdt()
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    eps_levels = sorted(df.eps.unique())
    xpos = {e: i for i, e in enumerate(eps_levels)}
    for j, (r, c, mk) in enumerate([(1.0, BLUES[0], "o"), (0.8, BLUES[1], "s"), (0.5, BLUES[2], "^")]):
        g = df[df.r == r].sort_values("eps")
        x = np.array([xpos[e] for e in g.eps]) + (j - 1) * 0.12
        y = g.S_star.astype(float).clip(lower=0.7)
        lo = g.S_star_lo.astype(float).clip(lower=0.7)
        hi = g.S_star_hi.astype(float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt=mk + "-", color=c, lw=0.8, ms=3.5,
                    label=fr"$r={r:g}$")
        zero = g.S_star_lo.values == 0
        if zero.any():
            ax.plot(x[zero], np.full(zero.sum(), 0.7), marker="v", ls="none", color=c, ms=3.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(eps_levels)), [f"{100 * e:g}" for e in eps_levels])
    ax.set_xlabel(r"Target miss $\varepsilon$ (%)")
    ax.set_ylabel(r"Largest admissible stride $S^*$ (frames)")
    ax.set_ylim(0.6, 160)
    ax.grid(True, which="major", axis="y", color=LIGHT, lw=0.4)
    sec = ax.secondary_yaxis("right", functions=(lambda s: F / np.maximum(s, 1e-9), lambda h: F / np.maximum(h, 1e-9)))
    sec.set_ylabel(f"Detector rate $F/S^*$ at {F} fps (Hz)")
    hz_ticks(sec, "y")
    ax.spines["right"].set_visible(True)
    ax.legend(loc="lower right", ncol=1, handlelength=1.6)
    fig.tight_layout()
    return save(fig, "fig_isomiss")


# ----------------------------------------------------------------------------- F3
def fig_onset() -> Path:
    d = json.loads((RES / "p0_stats.json").read_text(encoding="utf-8"))
    ro = d["p0b"]["rho_onset"]["UAVDT"]
    ws = d["p0b"]["meta"]["w_onset"]
    eps_list = sorted(d["p0b"]["meta"]["eps"])
    lam = ro["m0.00|w3"]["lambda_per_frame"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.9), gridspec_kw=dict(width_ratios=[1, 1.35]))
    # (a) rho_onset(w)
    pooled = np.array([ro[f"m0.00|w{w}"]["pooled"] for w in ws])
    ci = np.array([ro[f"m0.00|w{w}"]["pooled_ci"] for w in ws])
    p10 = np.array([ro[f"m0.00|w{w}"]["seq_p10"] for w in ws])
    p90 = np.array([ro[f"m0.00|w{w}"]["seq_p90"] for w in ws])
    a1.fill_between(ws, 100 * p10, 100 * p90, color=BLUES[3], lw=0, label="seq. p10–p90")
    a1.errorbar(ws, 100 * pooled, yerr=100 * np.vstack([pooled - ci[:, 0], ci[:, 1] - pooled]), fmt="o-",
                color=BLUES[0], ms=3, lw=0.9, label="pooled (CI)")
    wgrid = np.linspace(1, max(ws), 50)
    a1.plot(wgrid, 100 * lam * wgrid, ls="--", color=GREY, lw=0.8, label=r"$\lambda w$")
    a1.set_xlabel(r"Onset window $w$ (frames)")
    a1.set_ylabel(r"$\rho_{onset}(w)$ (% of frames)")
    a1.set_xticks(ws)
    a1.set_title("(a)", loc="left")
    a1.legend(loc="upper left", handlelength=1.4)
    # (b) activation at iso-KPI
    T = np.arange(2, 41)
    shown = [e for e in eps_list if e in (0.02, 0.05)] or eps_list[:2]  # main text: eps 2 %, 5 % (1 % -> appendix)
    for eps, c in zip(shown, [BLUES[0], BLUES[1]]):
        a2.plot(T, (1 - eps) / T, color=c, lw=1.0, label=fr"periodic $(1-\varepsilon)/T$, $\varepsilon={100 * eps:g}\%$")
    ylo = 0.5 * lam
    a2.axhline(lam, color=BLUES[2], lw=1.2, label=r"ideal onset gate $\geq\rho_1\approx\lambda$")
    a2.axhspan(ylo, lam, color=BLUES[3], alpha=0.6, lw=0)
    for w, ls in zip(ws, [":", "-.", (0, (1, 2))]):
        a2.axhline(ro[f"m0.00|w{w}"]["pooled"], color=GREY, lw=0.7, ls=ls)
        a2.text(T[-1] - 0.5, ro[f"m0.00|w{w}"]["pooled"] * 1.08, fr"$\rho_{{{w}}}$", va="bottom", ha="right", fontsize=8, color=GREY)
    a2.set_yscale("log")
    a2.set_ylim(ylo, 0.6)
    from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
    a2.yaxis.set_major_locator(FixedLocator([0.03, 0.05, 0.1, 0.2, 0.5]))
    a2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel(r"KPI window $T=L_{max}+1$ (frames)")
    a2.set_ylabel(r"Activation $a$ (fraction of frames)")
    a2.set_title("(b)", loc="left")
    a2.set_xlim(T[0], T[-1])
    a2.grid(True, which="major", color=LIGHT, lw=0.4)
    h, l = a2.get_legend_handles_labels()
    fig.tight_layout(w_pad=0.5, rect=(0, 0.2, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=8, handlelength=1.4, columnspacing=0.8,
               bbox_to_anchor=(0.5, 0.0))
    return save(fig, "fig_onset")


# ----------------------------------------------------------------------------- F4
def _heat(ax, mat, rows, cols, annot, title, xlabel, ylabel, ref=None):
    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax * 1.15, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            bold = ref is not None and v > ref
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, linespacing=0.95,
                    color="white" if v > 0.6 * vmax else "black", fontweight="bold" if bold else "normal")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


ISO_EPS_MAIN = 0.05  # main-text iso-KPI level for the theory illustration (eps = 1 % goes to the appendix, P2B-A2)
R_STYLE = {1.0: BLUES[0], 0.8: BLUES[1], 0.5: BLUES[2]}
G_MARK = {"low": "o", "mid": "s", "high": "^"}


def _qo_star_panel(ax, lax) -> tuple[int, int, int] | None:
    """(c) observed (replay) vs predicted q_o* with predicted CI, from results/p2/sim/mstar_iso.json."""
    f = SIM_DIR / "mstar_iso.json"
    if not f.exists():
        ax.text(0.5, 0.5, "mstar_iso.json absent", ha="center", va="center", transform=ax.transAxes)
        return None
    rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
    n_inf = n_deg = 0
    top = 0.0
    for x in rows:
        p, rp, ci = x["qo_star_pred"], x["qo_star_replay"], x.get("qo_star_pred_ci")
        if (isinstance(p, float) and math.isinf(p)) or (isinstance(rp, float) and math.isinf(rp)):
            n_inf += 1
            continue
        if p == 0 and ci is not None and ci[0] == 0 and ci[1] == 0:
            n_deg += 1
            continue
        c = R_STYLE.get(float(x["r"]), GREY)
        filled = abs(x["eps"] - 0.02) < 1e-9
        ax.errorbar(100 * p, 100 * rp, xerr=[[100 * (p - ci[0])], [100 * (ci[1] - p)]], fmt=G_MARK[x["group"]],
                    ms=4, color=c, mfc=c if filled else "white", mew=0.8, lw=0.6, capsize=1.2)
        top = max(top, 100 * ci[1], 100 * rp)
    top = 1.05 * top if top > 0 else 1
    ax.plot([0, top], [0, top], color=GREY, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(-0.2, top)
    ax.set_ylim(-0.2, top)
    ax.set_xlabel(r"Predicted $q_o^*$ (%), CI")
    ax.set_ylabel(r"Replay $q_o^*$ (%)")
    ax.set_title("(c) iso-KPI boundary, replay", loc="left")
    ax.grid(True, color=LIGHT, lw=0.4)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=R_STYLE[r], marker="o", ls="none", ms=4, label=fr"$r={r:g}$") for r in (1.0, 0.8, 0.5)]
    hs += [Line2D([], [], color=GREY, marker=G_MARK[g], ls="none", ms=4, mfc="white", label=f"{g} density") for g in ("low", "mid", "high")]
    hs += [Line2D([], [], color=GREY, marker="o", ls="none", ms=4, label=r"$\varepsilon=2\%$"),
           Line2D([], [], color=GREY, marker="o", ls="none", ms=4, mfc="white", label=r"$\varepsilon=5\%$")]
    lax.axis("off")
    lax.legend(handles=hs, loc="center left", handlelength=1.0, borderaxespad=0, labelspacing=0.3)
    return len(rows) - n_inf - n_deg, n_inf, n_deg


def fig_mstar() -> Path:
    t = json.loads((RES / "p1" / "theory_checks.json").read_text(encoding="utf-8"))
    P3 = t["P3_illustration"]
    Mmed = P3["M_median"]
    ms = pd.DataFrame(P3["M_star_table"])
    ms = ms[(ms.R.astype(str) == "inf") & (ms.q_b == 0.0)]
    Ls = sorted(ms.Lmax.unique())
    qos = sorted(ms.q_o.unique())
    mat = np.array([[float(ms[(ms.Lmax == L) & (ms.q_o == q)].M_star.iloc[0]) for q in qos] for L in Ls])
    ann = [[f"{int(ms[(ms.Lmax == L) & (ms.q_o == q)].M_star.iloc[0])}\n({ms[(ms.Lmax == L) & (ms.q_o == q)].M_nec.iloc[0]:.0f})"
            for q in qos] for L in Ls]
    iso = pd.DataFrame(t["isoKPI_illustration"]["table"])
    iso = iso[(iso.eps == ISO_EPS_MAIN) & (iso.c == 0.0) & (iso.q_b == 0.0)]
    Li = sorted(iso.Lmax.unique())
    qi = sorted(iso.q_o.unique())
    mat2 = np.array([[float(iso[(iso.Lmax == L) & (iso.q_o == q)].M_iso.iloc[0]) for q in qi] for L in Li])
    ann2 = [[f"{v:.0f}" for v in row] for row in mat2]
    fig = plt.figure(figsize=(COL_W, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[len(qos) + 0.4, len(qi) + 0.4],
                          hspace=0.55, wspace=0.45)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _heat(a1, mat, [str(L) for L in Ls], [f"{q:g}" for q in qos], ann,
          r"(a) $M^*$ ($M_{nec}$)", r"$q_o$", r"$L_{max}$ (frames)", ref=Mmed)
    _heat(a2, mat2, [str(L) for L in Li], [f"{q:g}" for q in qi], ann2,
          fr"(b) $M_{{iso}}$, $\varepsilon={100 * ISO_EPS_MAIN:g}\%$", r"$q_o$", "", ref=Mmed)
    a2.tick_params(axis="x", labelsize=8)
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.05)
    a3, lax = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    _qo_star_panel(a3, lax)
    return save(fig, "fig_mstar")


# ----------------------------------------------------------------------------- F5 (optional)
KPI_COLS = ["3", "5", "10", "miss"]
QOUT_ROWS = [0.0, 0.01, 0.02, 0.05, 0.1]
R_PANELS = [1.0, 0.8, 0.5]


def _best_cells(c: pd.DataFrame) -> dict:
    """Best Delta over (q_in, w, R) per (r, q_out, kpi); returns {(r, q_out, kpi): row}."""
    out = {}
    for (r, qo, k), g in c.groupby(["r", "q_out", c.kpi.astype(str)]):
        out[(float(r), float(qo), str(k))] = g.loc[g.delta.idxmax()]
    return out


def _g2prime_configs(c: pd.DataFrame, kpi: str = "5"):
    """Two representative configurations for Remark G2': largest rise and largest fall of Delta as r drops 1 -> 0.5."""
    g = c[c.kpi.astype(str) == kpi]
    piv = g.pivot_table(index=["q_in", "q_out", "w", "R"], columns="r", values="delta")
    if not {1.0, 0.5}.issubset(piv.columns):
        return []
    change = (piv[0.5] - piv[1.0]).dropna()
    if change.empty:
        return []
    return [change.idxmax(), change.idxmin()]


def fig_sim_map(sim_dir: Path = SIM_DIR) -> Path | None:
    """Gate vs periodic (matched cost), simulated detector: best Delta per (r, q_out, KPI) + Delta(r) for two configs."""
    f = sim_dir / "agnostic_cells.csv"
    if not f.exists():
        print("F5 skipped: results/p2/sim/agnostic_cells.csv absent")
        return None
    c = pd.read_csv(f)
    need = {"r", "fp", "q_in", "q_out", "w", "R", "event", "kpi", "M_bin", "delta", "lo", "hi"}
    if not need.issubset(c.columns):
        print("F5 skipped: unexpected schema", sorted(need - set(c.columns)))
        return None
    c = c[(c.event == "E3") & (c.M_bin == "all") & (c.fp == 0.0)].copy()
    c["R"] = c.R.astype(str)
    if c.empty:
        print("F5 skipped: no E3 / all-M / fp=0 cells")
        return None
    best = _best_cells(c)
    vmax = 100 * max(abs(v.delta) for v in best.values())
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(COL_W - 0.2, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], width_ratios=[1, 1, 1, 0.08], hspace=0.62, wspace=0.18)
    im = None
    for p, r in enumerate(R_PANELS):
        ax = fig.add_subplot(gs[0, p])
        mat = np.full((len(QOUT_ROWS), len(KPI_COLS)), np.nan)
        for i, qo in enumerate(QOUT_ROWS):
            for j, k in enumerate(KPI_COLS):
                row = best.get((r, qo, k))
                if row is None:
                    continue
                mat[i, j] = 100 * row.delta
                if row.lo > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.3, zorder=3))
                elif row.hi < 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=0, hatch="////", zorder=3))
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, zorder=4,
                            color="white" if abs(mat[i, j]) > 0.55 * vmax else "black")
        ax.set_xticks(range(len(KPI_COLS)), ["M" if k == "miss" else k for k in KPI_COLS])
        ax.set_yticks(range(len(QOUT_ROWS)), [f"{q:g}" for q in QOUT_ROWS] if p == 0 else [])
        if p == 0:
            ax.set_ylabel(r"$q_{out}$")
        if p == 1:
            ax.set_xlabel(r"KPI: $L_{max}$ (frames) or M = miss")
        ax.set_title(f"({'abc'[p]}) $r={r:g}$", loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cax.set_title(r"$\Delta$ (p.p.)", fontsize=8, loc="left", pad=4)
    cb.outline.set_linewidth(0.4)
    # (d) Remark G2'
    ax = fig.add_subplot(gs[1, :3])
    cfgs = _g2prime_configs(c, "5")
    for cfg, col, mk in zip(cfgs, [BLUES[0], BLUES[2]], ["o", "s"]):
        qi, qo, wv, R = cfg
        g = c[(c.kpi.astype(str) == "5") & (c.q_in == qi) & (c.q_out == qo) & (c.w == wv) & (c.R == R)].sort_values("r")
        Rl = r"\infty" if R == "inf" else f"{float(R):g}"
        ax.errorbar(g.r, 100 * g.delta, yerr=100 * np.vstack([g.delta - g.lo, g.hi - g.delta]), fmt=mk + "-",
                    color=col, ms=4, lw=0.9, capsize=1.5,
                    label=fr"$q_{{in}}={qi:g}$, $q_{{out}}={qo:g}$, $w={int(wv)}$, $R={Rl}$")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(R_PANELS)
    ax.invert_xaxis()
    ax.set_xlabel(r"Per-look recall $r$")
    ax.set_ylabel(r"$\Delta$ (p.p.), $L_{max}=5$")
    ax.set_title(r"(d) $\Delta$ vs $r$ (Remark G2′)", loc="left")
    ax.grid(True, color=LIGHT, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), handlelength=1.4, fontsize=8, ncol=1)
    return save(fig, "fig_sim_map")


def main() -> int:
    style()
    fig_exposure()
    fig_isomiss()
    fig_onset()
    fig_mstar()
    fig_sim_map()
    return 0


if __name__ == "__main__":
    sys.exit(main())
