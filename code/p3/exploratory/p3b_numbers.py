"""P3b E7 [POST HOC]: numbers.tex = p2_numbers + khối NumTest… (p3_numbers) + khối NumExp… (số trong RESULTS_28_EXPLORATORY.md).

Macro cũ không đổi tên/giá trị (p3_numbers kiểm trùng); tên chỉ chữ cái (p2_numbers.Macros). Chạy: python code/p3/exploratory/p3b_numbers.py
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code" / "p2"))
sys.path.insert(0, str(ROOT / "code" / "p3"))
import p2_numbers as P  # noqa: E402
import p3_numbers as P3  # noqa: E402

EX = ROOT / "results" / "p3" / "exploratory"
NAMES = ["e1_feasibility", "e2_recall_k", "e3_qostar", "e4_calibration", "e5_drift"]
EGO = {"hovering": "Hov", "moving": "Mov", "all": "All"}
MB = {"6-20": "SixToTwenty", ">20": "OverTwenty", "1-5": "OneToFive"}
CUE = {"border_band": "Border", "orb_lite_diff": "OrbLite", "raw_diff": "Raw", "tiny_det": "Tiny"}
LV = {"yolo26s_1024": "Main", "yolo26n_640": "Low"}
SPL = {"train": "Train", "test": "Test"}


def dec(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return P.NA
    if isinstance(x, float) and math.isinf(x):
        return "$\\infty$" if x > 0 else "$-\\infty$"
    return P.fmt_dec(x, nd)


def sdec(x, nd=3):
    s = dec(x, nd)
    return s if s.startswith("$") or s == P.NA or x < 0 else "+" + s


def J(n):
    return json.loads((EX / f"{n}.json").read_text(encoding="utf-8"))


def exp_macros(M):
    e1, e2, e3, e4, e5 = (J(n) for n in NAMES)
    M.sec("P3b POST HOC E1 -- iso-KPI feasibility (e1_feasibility.json)")
    for r in e1["strata"]:
        k = f"{EGO[r['ego']]}{MB[r['M_bin']]}L{P.w(r['L'])}"
        M.add(f"ExpFloorP{k}", P.fmt_pct(r["miss_P_floor_S1"]))
        M.add(f"ExpFloorG{k}", P.fmt_pct(r["gate_floor"]))
        M.add(f"ExpNEvents{k}", P.fmt_int(r["n_events"]))
        for cue, v in r["pred_a_P_iso_2pct"].items():
            M.add(f"ExpPredAPTwoPct{k}{CUE[cue]}", dec(v))
    for r in e1["iso"]:
        kind = "Feas" if r["eps_kind"] != "fixed" else P.w(round(r["eps"] * 100))
        k = f"ExpIso{EGO[r['ego']]}{MB[r['M_bin']]}L{P.w(r['L'])}Eps{kind}{CUE[r['cue']]}"
        if r["eps_kind"] != "fixed" and r["cue"] == "border_band":
            M.add(f"ExpEpsFeasPlusTwo{EGO[r['ego']]}{MB[r['M_bin']]}L{P.w(r['L'])}", P.fmt_pct(r["eps"]))
        M.add(k + "Diff", sdec(r["diff"]))
        M.add(k + "Lo", sdec(r["diff_lo"]) if r["diff_lo"] is not None else "$-\\infty$")
        M.add(k + "Hi", sdec(r["diff_hi"]) if r["diff_hi"] is not None else "$-\\infty$")
        M.add(k + "MissG", dec(r["miss_G"]))
        M.add(k + "AP", dec(r["a_P_iso"]))
    a, b = e1["share_correct_all"], e1["share_correct_excl_H2"]
    M.add("ExpCorrectAll", P.fmt_int(a["correct"]))
    M.add("ExpDecidedAll", P.fmt_int(a["decided"]))
    M.add("ExpShareCorrectAll", P.fmt_pct(a["share"]))
    M.add("ExpCorrectExclHTwo", P.fmt_int(b["correct"]))
    M.add("ExpDecidedExclHTwo", P.fmt_int(b["decided"]))
    M.add("ExpShareCorrectExclHTwo", P.fmt_pct(b["share"]))
    M.sec("P3b POST HOC E2 -- recall by frames since onset (e2_recall_k.json)")
    for lv, r in e2["r_model"].items():
        if lv in LV:
            M.add(f"ExpRModel{LV[lv]}", dec(r))
    for r in e2["rows"]:
        k = f"ExpRk{SPL[r['split']]}{LV[r['level']]}{r['e3_type'].capitalize()}K{P.w(r['k'])}"
        M.add(k, dec(r["r"]))
        if r["k"] <= 2:
            M.add(k + "CI", f"[{dec(r['lo'])}, {dec(r['hi'])}]")
    for r in e2["size"]:
        M.add(f"ExpSize{SPL[r['split']]}{r['e3_type'].capitalize()}K{P.w(r['k'])}", dec(r["sqrt_area_median"], 1))
    M.sec("P3b POST HOC E3 -- q_o* observed vs Thm P3-iii, descriptive (e3_qostar.json)")
    M.add("ExpQoRhoOneTrain", dec(e3["rho1_train"], 4))
    for r in e3["groups"]:
        k = f"ExpQo{r['group'].capitalize()}L{P.w(r['L'])}{'Frozen' if r['eps_kind'].startswith('frozen') else 'Feas'}"
        M.add(k + "Obs", dec(r["qo_star_obs"], 4))
        ci = r["qo_star_obs_ci"]
        M.add(k + "CI", P.NA if ci is None else f"[{dec(ci[0], 4)}, {dec(ci[1], 4)}]")
        M.add(k + "Pred", dec(r["qo_star_pred"], 4))
        M.add(k + "Eps", P.fmt_pct(r["eps"]))
        if k.endswith("Frozen"):
            M.add(f"ExpQo{r['group'].capitalize()}L{P.w(r['L'])}MMedian", P.fmt_int(r["M_median"]))
            M.add(f"ExpQo{r['group'].capitalize()}L{P.w(r['L'])}Floor", P.fmt_pct(r["miss_P_floor"]))
    M.sec("P3b POST HOC E4 -- calibration (e4_calibration.json)")
    for key, nm in (("all48", "All"), ("H3H5", "HThreeFive"), ("H8", "HEight")):
        c = e4[key]
        M.add(f"ExpCal{nm}N", P.fmt_int(c["n"]))
        M.add(f"ExpCal{nm}Slope", dec(c["slope"]))
        M.add(f"ExpCal{nm}Intercept", sdec(c["intercept"]))
        M.add(f"ExpCal{nm}MAE", dec(c["mae"]))
        M.add(f"ExpCal{nm}Bias", sdec(c["bias"]))
        M.add(f"ExpCal{nm}Spearman", dec(c["spearman"]))
    M.add("ExpCalSignAgree", P.fmt_pct(e4["sign_agree_point"]))
    u, ex = e4["H3H5_underestimate_loss"], e4["example_L5_border_hov_6_20"]
    M.add("ExpCalHThreeFiveBelowPred", P.fmt_int(u["n_obs_below_pred"]))
    M.add("ExpCalHThreeFiveMedianGap", sdec(u["median_obs_minus_pred"]))
    M.add("ExpCalExamplePred", sdec(ex["pred"]))
    M.add("ExpCalExampleObs", sdec(ex["obs"]))
    M.add("ExpCalExampleCI", f"[{sdec(ex['lo'])}, {sdec(ex['hi'])}]")
    M.sec("P3b POST HOC E5 -- channel drift TRAIN to TEST at frozen theta* (e5_drift.json)")
    for sp, s in e5["splits"].items():
        M.add(f"ExpLambda{SPL[sp]}", dec(s["lambda_per_frame"], 4))
        for w_, v in s["rho_onset"].items():
            M.add(f"ExpRho{P.w(int(w_))}{SPL[sp]}", dec(v, 4))
    for r in e5["cue_rows"]:
        k = f"ExpCh{CUE[r['cue']]}{EGO[r['ego']]}W{P.w(r['w'])}{SPL[r['split']]}"
        M.add(k + "QIn", dec(r["q_in"], 4))
        M.add(k + "QOut", dec(r["q_out"], 4))
        M.add(k + "J", sdec(r["J"], 4))
        M.add(k + "AG", dec(r["a_G_pred"], 4))
    for r in e5["strata_rows"]:
        k = f"ExpSt{CUE[r['cue']]}{EGO[r['ego']]}{MB[r['M_bin']]}{SPL[r['split']]}"
        M.add(k + "QIn", dec(r["q_in_stratum"], 4))
        M.add(k + "QOut", dec(r["q_out_ego"], 4))
        M.add(k + "J", sdec(r["q_in_stratum"] - r["q_out_ego"], 4))
        M.add(k + "PFire", dec(r["p_fire_any_k0_3"]))
    for r in e5["border_band_cells"]:
        k = f"ExpBb{EGO[r['ego']]}{MB[r['M_bin']]}L{P.w(int(r['kpi']))}"
        for f_, nm in (("miss_P_pred", "MissPPred"), ("miss_P_obs", "MissPObs"), ("miss_G_pred", "MissGPred"), ("miss_G_obs", "MissGObs"),
                       ("a_G_pred", "AGPred"), ("a_G_obs", "AGObs")):
            M.add(k + nm, dec(r[f_]))
        M.add(k + "DeltaPred", sdec(r["delta_pred"]))
        M.add(k + "DeltaObs", sdec(r["delta_obs"]))
        M.add(k + "DeltaCI", f"[{sdec(r['lo'])}, {sdec(r['hi'])}]")


def main():
    P3.main()
    txt = P.OUT.read_text(encoding="utf-8")
    old = set(re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", txt))
    M = P.Macros()
    exp_macros(M)
    dup = old & M.names
    assert not dup, sorted(dup)[:5]
    hdr = ("\n% ==== P3b POST HOC exploratory macros (NumExp...) -- GENERATED by code/p3/exploratory/p3b_numbers.py; NOT part of the frozen family\n"
           + "".join(f"% input: results/p3/exploratory/{n}.json  sha256[:16]={hashlib.sha256((EX / f'{n}.json').read_bytes()).hexdigest()[:16]}\n"
                     for n in NAMES))
    P.OUT.write_text(txt + M.render(hdr.rstrip("\n")), encoding="utf-8")
    print(f"appended {len(M.items)} NumExp macros; before {len(old)}")


if __name__ == "__main__":
    main()
