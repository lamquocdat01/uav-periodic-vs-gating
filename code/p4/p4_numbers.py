"""P4: numbers.tex for the manuscript = p2_numbers -> p3_numbers (NumTest) -> p3b_numbers (NumExp) -> P4 block, plus generated tables.

P4 block (appended; earlier macros keep their names and values):
  NumFz...   frozen TRAIN setup (channel_train.json, prereg_cells.json info, detector_recall_train.json, hover_threshold.json,
             audit_prebirth_train.json, bench_c_real.json, PREREG_28_FREEZE.md: tag, sha256, commit)
  NumCell... one block per confirmatory cell (results/p3/cells_test.csv; P3, not recomputed)
  NumDr...   first-order gate threshold J* = T c / (w' - T rho_w) (Cor. G4(iv), r factors out), TRAIN and TEST rho; J ratios
Tables (only macros inside; checked by p2_check_tex_numbers): paper/tables/tab_*.tex
Run: python code/p4/p4_numbers.py
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
sys.path.insert(0, str(ROOT / "code" / "p3"))
sys.path.insert(0, str(ROOT / "code" / "p3" / "exploratory"))
import p2_numbers as P  # noqa: E402
import p3b_numbers as P3B  # noqa: E402

R2, R3, P0 = ROOT / "results" / "p2", ROOT / "results" / "p3", ROOT / "results" / "p0"
TAB = ROOT / "paper" / "tables"
CUE = {"border_band": "Border", "orb_lite_diff": "OrbLite", "raw_diff": "Raw", "tiny_det": "Tiny", "ego_comp_diff": "EgoComp"}
CUE_TEX = {"border_band": "border band", "orb_lite_diff": "ORB-lite diff.", "raw_diff": "raw diff.", "tiny_det": "tiny det."}
EGO = {"hovering": "Hov", "moving": "Mov"}
MB = {"6-20": "SixToTwenty", ">20": "OverTwenty"}
MB_TEX = {"6-20": "6--20", ">20": "$>$20"}  # labels only; digits here are table labels -> written via macros below
VERD = {"correct": "correct", "wrong": "wrong", "inconclusive": "inconcl."}
SRC = [R2 / "channel_train.json", R2 / "prereg_cells.json", R2 / "detector_recall_train.json", P0 / "hover_threshold.json",
       P0 / "audit_prebirth_train.json", R2 / "bench_real" / "bench_c_real.json", ROOT / "PREREG_28_FREEZE.md", R3 / "cells_test.csv",
       R3 / "exploratory" / "e5_drift.json"]


def dec(x, nd=3):
    return P3B.dec(x, nd)


def sdec(x, nd=3):
    return P3B.sdec(x, nd)


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cell_key(x):
    g = {"H2": "HTwo", "H3-H5": "HThree", "H8": "HEight"}[x["group"]]
    L = P.w(int(str(x["kpi"]).split("_")[-1].lstrip("L")))
    return f"Cell{g}L{L}{EGO[x['ego']]}{MB[x['M_bin']]}{CUE[x['cue']]}"


def fz_macros(M):
    ch = J(R2 / "channel_train.json")
    info = J(R2 / "prereg_cells.json")["info"]
    cells = J(R2 / "prereg_cells.json")["cells"]
    conf = [c for c in cells if c["confirmatory"]]
    M.sec("P4 frozen TRAIN setup (channel_train.json, prereg_cells.json, PREREG_28_FREEZE.md)")
    frz = (ROOT / "PREREG_28_FREEZE.md").read_text(encoding="utf-8")
    M.add("FzTag", "\\texttt{prereg-28-v1}")
    M.add("FzPreregFile", "\\texttt{PREREG\\_28.md}")
    sha = re.search(r"`PREREG_28.md` \| `([0-9a-f]{64})`", frz).group(1)
    M.add("FzPreregSha", "\\texttt{" + "\\allowbreak{}".join(sha[i:i + 8] for i in range(0, 64, 8)) + "}")  # breakable in two columns
    M.add("FzPreregShaShort", "\\texttt{" + re.search(r"`PREREG_28.md` \| `([0-9a-f]{64})`", frz).group(1)[:16] + "}")
    M.add("FzCellsShaShort", "\\texttt{" + re.search(r"`results/p2/prereg_cells.json` \| `([0-9a-f]{64})`", frz).group(1)[:16] + "}")
    M.add("FzCommit", "\\texttt{5cfe901}")
    M.add("FzDate", "1~October~2026")
    M.add("FzTestDate", "3~October~2026")
    for k, v in (("PTwoB", "27~September"), ("PTwoC", "28~September"), ("PTwoE", "30~September"), ("PTwoF", "1~October (morning)"),
                 ("PTwoG", "1~October (afternoon)"), ("PTwoH", "1~October (evening)")):  # LOG_28 headers of each step
        M.add(f"FzDate{k}", v)
    M.add("FzFreezeTime", "17{:}04 (UTC$+$7)")
    M.add("FzUltralytics", "8.4.163")
    M.add("FzOpenVINO", "2026.4")
    M.add("FzPython", "3.11.9")
    M.add("FzTrainSeq", P.fmt_int(ch["n_seq"]))
    M.add("FzTrainFrames", P.fmt_int(ch["n_frames"]))
    M.add("FzTrainEThree", P.fmt_int(ch["n_e3_train"]))
    M.add("FzLambda", dec(ch["lambda_per_frame"], 4))
    for w_, v in ch["rho_onset"].items():
        M.add(f"FzRhoW{P.w(int(w_))}", dec(v, 3))
    M.add("FzHoverThr", dec(ch["hover_threshold_px"], 2))
    hv = J(P0 / "hover_threshold.json")["by_dataset"]["UAVDT"]["gmm2"]
    M.add("FzHoverModeLo", dec(hv["mu_px"][0], 2))
    M.add("FzHoverModeHi", dec(hv["mu_px"][1], 2))
    M.add("FzHoverAshman", dec(hv["ashman_D"], 2))
    M.add("FzSeqHovering", P.fmt_int(ch["ego_counts"]["hovering"]))
    M.add("FzSeqMoving", P.fmt_int(ch["ego_counts"]["moving"]))
    M.add("FzScoreMain", dec(ch["score_min_main"], 2))
    M.add("FzScoreLo", dec(0.05, 2))
    M.add("FzScoreHi", dec(0.50, 2))
    M.add("FzIoU", dec(0.5, 1))
    rec = J(R2 / "detector_recall_train.json")["levels"]
    for lv, nm in (("yolo26s_1024", "SOneThousandTwentyFour"), ("yolo26n_1024", "NOneThousandTwentyFour"), ("yolo26s_640", "SSixHundredForty"),
                   ("yolo26n_640", "NSixHundredForty")):
        x = rec[lv]["score>=0.25"]
        M.add(f"FzR{nm}", dec(x["r"], 3))
        M.add(f"FzR{nm}CI", f"[{dec(x['ci'][0], 3)}, {dec(x['ci'][1], 3)}]")
        M.add(f"FzRGap{nm}", dec(ch["r_levels"]["yolo26s_1024"] - x["r"], 3))
    M.add("FzHEightDelta", dec(info["h8_rule"]["delta"], 2))
    M.add("FzImgMain", "1024")
    M.add("FzImgLow", "640")
    M.add("FzImgTiny", "320")
    M.add("FzR", P.fmt_int(info["conf_R"]))
    M.add("FzW", P.fmt_int(info["w_star"]))
    M.add("FzAMax", dec(info["a_max"], 2))
    M.add("FzDeltaMin", dec(0.005, 3))
    M.add("FzEpsMain", P.fmt_pct(0.02, 0))
    M.add("FzEpsDropped", P.fmt_pct(0.05, 0))
    M.add("FzFamilyMax", P.fmt_int(info["family"]["fmax"]))
    for s in info["family"]["log"]:
        nm = {"start": "Start", "drop_eps5": "DropEpsFive", "drop_L10": "DropLTen", "H8_rdagger": "HEightRDagger"}[s["step"]]
        M.add("FzFamily" + nm, P.fmt_int(s["n"]))
    M.add("FzFamilyPlus", P.fmt_int(sum(c["sign_pred"] == "+" for c in conf)))
    M.add("FzFamilyPlusHThree", P.fmt_int(sum(c["sign_pred"] == "+" and c["hypothesis"] == "H3/H5" for c in conf)))
    M.add("FzFamilyPlusHEight", P.fmt_int(sum(c["sign_pred"] == "+" and c["hypothesis"] == "H8" for c in conf)))
    M.add("FzCellMinEvents", P.fmt_int(30))
    M.add("FzCellMinSeq", P.fmt_int(3))
    M.add("FzPowerMax", P.fmt_pct(0.5, 0))
    M.add("FzPassShare", P.fmt_pct(0.8, 0))
    M.add("FzHEightWrongMax", P.fmt_pct(0.5, 0))
    rd = [c["r_dagger"] for c in cells if c.get("r_dagger") is not None]
    M.add("FzRDaggerMax", dec(max(rd), 3))
    # strata TRAIN counts
    for (eg, mb) in (("hovering", "6-20"), ("hovering", ">20"), ("moving", "6-20")):
        c = next(c for c in conf if c["ego"] == eg and c["M_bin"] == mb)
        M.add(f"FzN{EGO[eg]}{MB[mb]}", P.fmt_int(c["n_events_train"]))
        M.add(f"FzN{EGO[eg]}{MB[mb]}Seq", P.fmt_int(c["n_seq_train"]))
    # theta* per cue
    for s in info["theta_star"]:
        k = f"FzCue{CUE[s['cue']]}"
        M.add(k + "C", dec(s["c"], 3))
        if s["theta"] is None:
            continue
        M.add(k + "Theta", dec(s["theta"], 4))
        M.add(k + "QIn", dec(s["q_in"], 3))
        M.add(k + "QOut", dec(s["q_out"], 3))
        M.add(k + "J", dec(s["J"], 3))
        M.add(k + "JFree", dec(s["J_unconstrained"], 3))
        M.add(k + "AG", dec(s["a_G"], 3))
    js = [s["J"] for s in info["theta_star"] if s["J"] is not None]
    M.add("FzJMax", dec(max(js), 3))
    M.add("FzJMin", dec(min(js), 3))
    ops = {o["cue"]: o for o in ch["operating_points"] if o.get("theta_star")}
    for cue, o in ops.items():
        for eg in ("hovering", "moving"):
            M.add(f"FzCue{CUE[cue]}QB{EGO[eg]}", dec(o["q_b"][eg], 3))
    sess = [s["det_ms"] for s in ch["det_ms_sessions"]]
    M.add("FzDetMsLo", dec(min(sess), 1))
    M.add("FzDetMsHi", dec(max(sess), 1))
    M.add("FzDetMsC", dec(ch["det_ms_sessions"][-1]["det_ms"], 1))
    b = J(R2 / "bench_real" / "bench_c_real.json")
    M.add("FzBenchPairs", P.fmt_int(b["n_pairs"]))
    a = J(P0 / "audit_prebirth_train.json")
    M.add("FzAuditInteriorLate", P.fmt_pct(a["table_main"]["interior"]["share_annotation_late"]))
    M.add("FzAuditBorderLate", P.fmt_pct(a["table_main"]["border"]["share_annotation_late"]))
    M.add("FzAuditPlacebo", P.fmt_pct(a["placebo"]["hit_rate"]))
    M.add("FzOnsetAltShifted", P.fmt_int(a["onset_alt"]["n_shifted"]))
    M.add("FzOnsetAltShift", dec(a["onset_alt"]["shift_mean"], 1))
    ch_alt = [c["delta_pred_onset_alt"] - c["delta_pred"] for c in conf if c.get("delta_pred_onset_alt") is not None]
    M.add("FzOnsetAltMaxChange", dec(max(abs(x) for x in ch_alt), 4))
    M.add("FzOnsetAltSignChanges", P.fmt_int(sum((c["delta_pred_onset_alt"] > 0.005) != (c["delta_pred"] > 0.005) for c in conf
                                                  if c.get("delta_pred_onset_alt") is not None)))
    # withdrawn P2F version (LOG_28): share of TRAIN E3 removed
    M.add("FzPTwoFRemoved", P.fmt_pct(1 - 153 / 865, 0))
    M.add("FzPTwoFKept", P.fmt_int(153))


def cell_macros(M):
    d = pd.read_csv(R3 / "cells_test.csv")
    M.sec("P4 per-cell values of the confirmatory family (results/p3/cells_test.csv, P3; not recomputed)")
    for x in d.to_dict("records"):
        k = cell_key(x)
        M.add(k + "Pred", "$+$" if x["sign_pred"] == "+" else "$\\le 0$")
        M.add(k + "Verdict", VERD[x["verdict"]])
        M.add(k + "N", P.fmt_int(x["n_events_test"]))
        M.add(k + "NSeq", P.fmt_int(x["n_seq_test"]))
        if x["group"] == "H2":
            M.add(k + "PredAG", dec(x["pred_a_G"]))
            M.add(k + "PredAP", dec(x["pred_a_P_iso"]))
            M.add(k + "ObsAG", dec(x["a_G"]))
            M.add(k + "ObsMissG", dec(x["miss_G"]))
            M.add(k + "ObsAP", "$\\infty$" if math.isinf(x["a_P_iso"]) else dec(x["a_P_iso"]))
            M.add(k + "DiffHi", sdec(x["hi"]))
        else:
            M.add(k + "PredVal", sdec(x["pred_value"], 3))
            M.add(k + "Obs", sdec(x["obs_value"], 3))
            M.add(k + "Lo", sdec(x["lo"], 3))
            M.add(k + "Hi", sdec(x["hi"], 3))
    pl = d[(d.sign_pred == "+") & (d.group == "H3-H5")]
    M.add("CellHThreePlusN", P.fmt_int(len(pl)))
    win = d[(d.group == "H3-H5") & (d.lo > 0)]
    M.add("CellHThreeGateWinsCI", P.fmt_int(len(win)))
    h8 = d[d.group == "H8"]
    M.add("CellHEightObsPos", P.fmt_int((h8.obs_value > 0).sum()))
    M.add("CellHEightLoPos", P.fmt_int((h8.lo > 0).sum()))
    M.add("CellHEightObsMin", sdec(h8.obs_value.min(), 3))
    M.add("CellHEightPlusN", P.fmt_int((h8.sign_pred == "+").sum()))
    M.add("CellHEightPlusCorrect", P.fmt_int(((h8.sign_pred == "+") & (h8.verdict == "correct")).sum()))
    M.add("CellHEightPlusInconcl", P.fmt_int(((h8.sign_pred == "+") & (h8.verdict == "inconclusive")).sum()))
    M.add("CellHEightLeZeroN", P.fmt_int((h8.sign_pred != "+").sum()))
    M.add("CellHEightLeZeroWrong", P.fmt_int(((h8.sign_pred != "+") & (h8.verdict == "wrong")).sum()))
    M.add("CellHEightLeZeroInconcl", P.fmt_int(((h8.sign_pred != "+") & (h8.verdict == "inconclusive")).sum()))
    M.add("CellHThreeLeZeroCorrect", P.fmt_int(((d.group == "H3-H5") & (d.sign_pred != "+") & (d.verdict == "correct")).sum()))
    M.add("CellHThreeLeZeroN", P.fmt_int(((d.group == "H3-H5") & (d.sign_pred != "+")).sum()))
    M.add("CellHThreeLeZeroInconcl", P.fmt_int(((d.group == "H3-H5") & (d.sign_pred != "+") & (d.verdict == "inconclusive")).sum()))
    hl = d[d.group == "H3-H5"]
    M.add("CellHThreeObsMax", sdec(hl.obs_value.max(), 3))
    M.add("CellHThreeHiMax", sdec(hl.hi.max(), 3))


def dr_macros(M):
    """First-order threshold (Cor. G4(iv) with r factored out): Delta ~ r[(q_in - q_out)(w' - T rho_w) - T c] > 0 iff J > J*."""
    ch = J(R2 / "channel_train.json")
    info = J(R2 / "prereg_cells.json")["info"]
    e5 = J(R3 / "exploratory" / "e5_drift.json")
    rho5 = {"train": ch["rho_onset"]["5"], "test": e5["splits"]["test"]["rho_onset"]["5"]}
    M.sec("P4 design rule: first-order gate threshold J* = T c/(w' - T rho_w), w = 5 (Cor. G4(iv)); J ratio TRAIN/TEST (e5_drift.json)")
    jtest = {r["cue"]: r["J"] for r in e5["cue_rows"] if r["w"] == 5 and r["ego"] == "all" and r["split"] == "test"}
    ratios = []
    for s in info["theta_star"]:
        if s["theta"] is None:
            continue
        k = CUE[s["cue"]]
        for L in (3, 5):
            T, wp = L + 1, min(5, L + 1)
            for sp, rr in rho5.items():
                js = T * s["c"] / (wp - T * rr)
                M.add(f"DrJStar{k}L{P.w(L)}{sp.capitalize()}", dec(js, 3))
        rt = s["J"] / jtest[s["cue"]]
        ratios.append(rt)
        M.add(f"DrJRatio{k}", dec(rt, 1))
    M.add("DrJRatioMin", dec(min(ratios), 1))
    M.add("DrJRatioMax", dec(max(ratios), 1))
    M.add("DrJTestMax", dec(max(jtest.values()), 3))
    for cue, v in jtest.items():
        M.add(f"DrJTest{CUE[cue]}", dec(v, 3))
    M.add("DrRhoFiveTest", dec(rho5["test"], 3))


# ------------------------------------------------------------------ tables (macros only)
def m(name):
    return "\\Num" + name + "{}"


def tables():
    TAB.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(R3 / "cells_test.csv")
    cues = ["border_band", "orb_lite_diff", "raw_diff", "tiny_det"]
    # T1 frozen setup
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Frozen operating points (UAVDT TRAIN, \\NumFzTrainSeq{} sequences, \\NumFzTrainEThree{} onset events). "
         "$\\theta^*$ maximises $J=q_{in}-q_{out}$ at $w=\\NumFzW$ subject to $a_G(R=\\NumFzR)+c\\le\\NumFzAMax$; "
         "$J_{free}$: maximum without the constraint; $c$: cue cost in detector equivalents (same benchmark session); "
         "$q_b$: background firing per ego class. Pre-registered (tag \\NumFzTag).}",
         "\\label{tab:frozen}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.0pt}", "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
         "cue & $\\theta^*$ & $q_{in}$ & $q_{out}$ & $J$ & $J_{free}$ & $a_G$ & $c$ & $q_b$ hov./mov. \\\\", "\\midrule"]
    for c in cues:
        k = "FzCue" + CUE[c]
        L.append(f"{CUE_TEX[c]} & {m(k + 'Theta')} & {m(k + 'QIn')} & {m(k + 'QOut')} & {m(k + 'J')} & {m(k + 'JFree')} & {m(k + 'AG')} & "
                 f"{m(k + 'C')} & {m(k + 'QBHov')}/{m(k + 'QBMov')} \\\\")
    L += [f"ORB diff.\\ (full) & \\multicolumn{{6}}{{c}}{{excluded: $c\\ge 1$ (Thm.~\\ref{{thm:g3}})}} & {m('FzCueEgoCompC')} & -- \\\\", "\\bottomrule",
          "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_frozen.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # T2 family
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Confirmatory family on UAVDT TEST (pre-registered, \\NumTestFamily{} cells, three-way scoring). "
         "Last row (post hoc): the \\NumTestHTwoN{} iso-KPI cells (H2) removed because neither policy can reach $\\varepsilon=\\NumFzEpsMain$ (Sec.~\\ref{sec:res-mech}).}",
         "\\label{tab:family}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         "group & cells & correct & wrong & inconcl. & correct/decided \\\\", "\\midrule",
         f"H2 (iso-KPI) & {m('TestHTwoN')} & {m('TestHTwoNCorrect')} & {m('TestHTwoNWrong')} & {m('TestHTwoNInconclusive')} & -- \\\\",
         f"H3/H5 (matched cost) & {m('TestHThreeFiveN')} & {m('TestHThreeFiveNCorrect')} & {m('TestHThreeFiveNWrong')} & {m('TestHThreeFiveNInconclusive')} & -- \\\\",
         f"H8 (recall pair) & {m('TestHEightN')} & {m('TestHEightNCorrect')} & {m('TestHEightNWrong')} & {m('TestHEightNInconclusive')} & -- \\\\", "\\midrule",
         f"all & {m('TestFamily')} & {m('TestNCorrect')} & {m('TestNWrong')} & {m('TestNInconclusive')} & {m('ExpShareCorrectAll')} \\\\",
         f"without H2 (post hoc) & {m('ExpCalAllN')} & {m('ExpCorrectExclHTwo')} & {m('TestNWrong')} & {m('TestNInconclusive')} & {m('ExpShareCorrectExclHTwo')} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_family.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # T3 plus cells
    pl = d[d.sign_pred == "+"].sort_values(["group", "kpi", "cue", "ego", "M_bin"])
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{The \\NumTestPlusN{} cells in which the frozen model predicted a gate advantage ($\\Delta_{pred}>\\delta_{min}$), UAVDT TEST "
         "(pre-registered). H3/H5: $\\hat\\Delta=\\miss_P-\\miss_G$ at matched cost; H8: $\\widehat{\\Delta\\Delta}=\\hat\\Delta(\\text{yolo26n@}\\NumFzImgLow)-\\hat\\Delta(\\text{yolo26s@}\\NumFzImgMain)$; "
         "\\NumCIPct{} paired sequence-bootstrap CI.}", "\\label{tab:plus}", "\\scriptsize", "\\setlength{\\tabcolsep}{1.6pt}",
         "\\begin{tabular}{@{}llllcccl@{}}", "\\toprule", "hyp. & $\\Lmax$ & cue & stratum & pred. & obs. & CI & result \\\\", "\\midrule"]
    for x in pl.to_dict("records"):
        k = cell_key(x)
        Lm = "\\NumLmaxLThree" if str(x["kpi"]).endswith("3") else "\\NumLmaxLFive"
        st = ("hov." if x["ego"] == "hovering" else "mov.") + " " + ("\\NumStratSixTwenty" if x["M_bin"] == "6-20" else "\\NumStratOverTwenty")
        L.append(f"{'H3/H5' if x['group'] == 'H3-H5' else 'H8'} & {Lm} & {CUE_TEX[x['cue']]} & {st} & {m(k + 'PredVal')} & {m(k + 'Obs')} & "
                 f"[{m(k + 'Lo')}, {m(k + 'Hi')}] & {m(k + 'Verdict')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_plus.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # T4 calibration
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Calibration of the frozen mean-field predictions against UAVDT TEST (post hoc): ordinary least squares of the observed "
         "on the predicted value, mean absolute error, mean bias (observed $-$ predicted) and Spearman rank correlation.}",
         "\\label{tab:calib}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "cells & $n$ & slope & intercept & MAE & bias & Spearman \\\\", "\\midrule"]
    for k, lab in (("HThreeFive", "H3/H5 ($\\Delta$)"), ("HEight", "H8 ($\\Delta\\Delta$)"), ("All", "both")):
        b = "ExpCal" + k
        L.append(f"{lab} & {m(b + 'N')} & {m(b + 'Slope')} & {m(b + 'Intercept')} & {m(b + 'MAE')} & {m(b + 'Bias')} & {m(b + 'Spearman')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_calib.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # T5 variants
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Descriptive variants of the \\NumTestFamily{} confirmatory cells on UAVDT TEST (not part of the criteria; H2 cells scored by sign only).}",
         "\\label{tab:variants}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "variant & correct & wrong & inconcl. \\\\", "\\midrule",
         f"frozen protocol (Table~\\ref{{tab:family}}) & {m('TestNCorrect')} & {m('TestNWrong')} & {m('TestNInconclusive')} \\\\"]
    for vn, lab in (("OnsetAlt", "earlier onset (onset$_{alt}$)"), ("RoiFive", "ROI margin \\NumRoiMarginFive"),
                    ("ScorePtZeroFive", "detector score $\\ge\\NumFzScoreLo$"), ("ScorePtFifty", "detector score $\\ge\\NumFzScoreHi$")):
        L.append(f"{lab} & {m('TestSens' + vn + 'Correct')} & {m('TestSens' + vn + 'Wrong')} & {m('TestSens' + vn + 'Inconclusive')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_variants.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # Appendix: 72 cells, three tables
    for grp, cap, lab in (("H2", "iso-KPI cells (H2), $\\varepsilon=\\NumFzEpsMain$: predicted $a_G$ / $a_P(\\varepsilon)$ versus observed; "
                                 "$a_P(\\varepsilon)=\\infty$: no periodic stride reaches $\\varepsilon$; CI$_{hi}$: upper \\NumCIPct{} limit of $a_G-a_P(\\varepsilon)$.", "tab:cellsH2"),
                          ("H3-H5", "matched-cost cells (H3/H5): $\\Delta_{pred}$ and $\\hat\\Delta$ with \\NumCIPct{} CI.", "tab:cellsH3"),
                          ("H8", "recall-pair cells (H8): $\\Delta\\Delta_{pred}$ and $\\widehat{\\Delta\\Delta}$ with \\NumCIPct{} CI.", "tab:cellsH8")):
        g = d[d.group == grp].sort_values(["cue", "ego", "M_bin", "kpi"])
        env = "table*" if grp == "H2" else "table"  # H2 has ten columns -> double column
        L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", f"\\begin{{{env}}}[t]", "\\centering",
             f"\\caption{{UAVDT TEST, pre-registered {cap}}}", f"\\label{{{lab}}}", "\\footnotesize" if grp == "H2" else "\\scriptsize",
             "\\setlength{\\tabcolsep}{4pt}" if grp == "H2" else "\\setlength{\\tabcolsep}{2pt}"]
        if grp == "H2":
            L += ["\\begin{tabular}{@{}llcccccccl@{}}", "\\toprule",
                  "cue & stratum & $\\Lmax$ & pred. & $a_G$ pr. & $a_P$ pr. & $a_G$ obs. & $\\miss_G$ & CI$_{hi}$ & result \\\\", "\\midrule"]
        else:
            L += ["\\begin{tabular}{@{}llccccl@{}}", "\\toprule", "cue & stratum & $\\Lmax$ & pred. & obs. & CI & result \\\\", "\\midrule"]
        for x in g.to_dict("records"):
            k = cell_key(x)
            Lm = "\\NumLmaxLThree" if str(x["kpi"]).endswith("3") else "\\NumLmaxLFive"
            st = ("hov." if x["ego"] == "hovering" else "mov.") + " " + ("\\NumStratSixTwenty" if x["M_bin"] == "6-20" else "\\NumStratOverTwenty")
            if grp == "H2":
                L.append(f"{CUE_TEX[x['cue']]} & {st} & {Lm} & {m(k + 'Pred')} & {m(k + 'PredAG')} & {m(k + 'PredAP')} & {m(k + 'ObsAG')} & "
                         f"{m(k + 'ObsMissG')} & {m(k + 'DiffHi')} & {m(k + 'Verdict')} \\\\")
            else:
                L.append(f"{CUE_TEX[x['cue']]} & {st} & {Lm} & {m(k + 'PredVal')} & {m(k + 'Obs')} & [{m(k + 'Lo')}, {m(k + 'Hi')}] & {m(k + 'Verdict')} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
        (TAB / f"tab_cells_{grp.replace('-', '')}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # Appendix: drift (E5)
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Channel drift at the frozen $\\theta^*$, $w=\\NumFzW$ (post hoc): onset-window separation $J=q_{in}-q_{out}$ on TRAIN and TEST, "
         "all sequences and per stratum (stratum $q_{in}$ uses only the onset windows of the stratum's events). First-order threshold $J^*$ "
         "for $\\Lmax=\\NumLmaxLThree$ (Sec.~VII-A of the main text).}", "\\label{tab:drift}", "\\scriptsize", "\\setlength{\\tabcolsep}{1.7pt}",
         "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{all} & \\multicolumn{2}{c}{hov.\\ \\NumStratSixTwenty} & \\multicolumn{2}{c}{hov.\\ \\NumStratOverTwenty} & "
         "\\multicolumn{2}{c}{$J^*$, $\\Lmax=\\NumLmaxLThree$} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "cue & TRAIN & TEST & TRAIN & TEST & TRAIN & TEST & TRAIN & TEST \\\\", "\\midrule"]
    for c in cues:
        k = CUE[c]
        L.append(f"{CUE_TEX[c]} & {m(f'ExpCh{k}AllWFiveTrainJ')} & {m(f'ExpCh{k}AllWFiveTestJ')} & "
                 f"{m(f'ExpSt{k}HovSixToTwentyTrainJ')} & {m(f'ExpSt{k}HovSixToTwentyTestJ')} & {m(f'ExpSt{k}HovOverTwentyTrainJ')} & "
                 f"{m(f'ExpSt{k}HovOverTwentyTestJ')} & {m(f'DrJStar{k}LThreeTrain')} & {m(f'DrJStar{k}LThreeTest')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_drift.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # Appendix: feasibility floors (E1)
    L = ["% GENERATED by code/p4/p4_numbers.py -- do not edit", "\\begin{table}[t]", "\\centering",
         "\\caption{Latency-miss floors on UAVDT TEST (post hoc): periodic schedule running the detector on every frame ($S=1$) and the best gate over "
         "the TRAIN threshold grid ($R=\\NumFzR$), yolo26s@\\NumFzImgMain. A target $\\varepsilon$ below the periodic floor cannot be met by any schedule.}",
         "\\label{tab:floors}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{$\\Lmax=\\NumLmaxLThree$} & \\multicolumn{2}{c}{$\\Lmax=\\NumLmaxLFive$} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "stratum (events) & periodic & gate & periodic & gate \\\\", "\\midrule"]
    for eg, mb in (("hovering", "6-20"), ("hovering", ">20"), ("moving", "6-20")):
        b = f"{EGO[eg]}{MB[mb]}"
        st = ("hov." if eg == "hovering" else "mov.") + " " + ("\\NumStratSixTwenty" if mb == "6-20" else "\\NumStratOverTwenty")
        L.append(f"{st}~({m('ExpNEvents' + b + 'LThree')}) & {m('ExpFloorP' + b + 'LThree')} & {m('ExpFloorG' + b + 'LThree')} & "
                 f"{m('ExpFloorP' + b + 'LFive')} & {m('ExpFloorG' + b + 'LFive')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_floors.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def label_macros(M):
    M.sec("P4 table labels")
    M.add("StratSixTwenty", "$M$\\,6--20")
    M.add("StratOverTwenty", "$M\\!>\\!20$")
    M.add("StratOneFive", "$M$\\,1--5")


def main():
    P3B.main()
    txt = P.OUT.read_text(encoding="utf-8")
    old = set(re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", txt))
    M = P.Macros()
    fz_macros(M)
    cell_macros(M)
    dr_macros(M)
    label_macros(M)
    dup = old & M.names
    assert not dup, sorted(dup)[:5]
    hdr = ("\n% ==== P4 manuscript macros (NumFz frozen setup, NumCell per cell, NumDr design rule) -- GENERATED by code/p4/p4_numbers.py\n"
           + "".join(f"% input: {p.relative_to(ROOT).as_posix()}  sha256[:16]={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}\n" for p in SRC))
    P.OUT.write_text(txt + M.render(hdr.rstrip("\n")), encoding="utf-8")
    tables()
    print(f"appended {len(M.items)} P4 macros; before {len(old)}; tables -> {TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
