"""Generate paper/numbers.tex: every data number used in the paper, as a LaTeX macro.

Inputs (read-only):
  results/p0_stats.json            P0 / P0b statistics (UAVDT)
  results/p1/theory_checks.json    numerical checks + illustrations of THEORY_28
  code/theory_checks.py            only to read the illustration constants (h, theta, v_max, eps, kappa, s0, r_max)
  results/p2/sim/summary.json      OPTIONAL (simulated-detector replay); skipped if absent

Output: paper/numbers.tex  (\\newcommand{\\NumXxx}{...}, letters-only names).

Formatting rules (one place, used everywhere):
  shares / probabilities   -> percent with one decimal + '\\%'           fmt_pct
  counts                   -> integer, thousands separated by '{,}'     fmt_int
  strides S*, n, M*        -> integer                                   fmt_int
  rates in Hz              -> 2 decimals if < 1, else 1 decimal         fmt_hz
  activation a             -> 3 decimals                                fmt_dec(x, 3)
  other reals              -> explicit number of decimals               fmt_dec
  CI                       -> separate ...Lo / ...Hi macros and a ...CI macro '[lo, hi]'
Use macros in text as \\NumXxx{} (no xspace).

Run: .venv/Scripts/python.exe code/p2/p2_numbers.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P0 = ROOT / "results" / "p0_stats.json"
P1 = ROOT / "results" / "p1" / "theory_checks.json"
THEORY_PY = ROOT / "code" / "theory_checks.py"
SIM = ROOT / "results" / "p2" / "sim" / "summary.json"
OUT = ROOT / "paper" / "numbers.tex"

# ----------------------------------------------------------------------------- formatting
ONES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
        "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]


def w(n: int) -> str:
    """Integer -> CamelCase English words (letters only)."""
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + "Hundred" + ("" if n % 100 == 0 else w(n % 100))
    raise ValueError(n)


def wdec(x: float) -> str:
    """0.8 -> PtEight, 1.0 -> One, 0.005 -> PtZeroZeroFive."""
    if float(x).is_integer():
        return w(int(x))
    s = f"{x:g}"
    ip, fp = s.split(".")
    return (w(int(ip)) if ip != "0" else "") + "Pt" + "".join(ONES[int(c)] for c in fp)


def wpct(x: float) -> str:
    """0.05 -> FivePct, 0.10 -> TenPct, 0.005 -> PtFivePct."""
    p = round(x * 100, 6)
    return (w(int(p)) if float(p).is_integer() else wdec(p)) + "Pct"


NA = "--"  # rendered for undefined values (e.g. Hz of an infeasible S* = 0)


def fmt_pct(x: float, nd: int = 1) -> str:
    if x is None:
        return NA
    return f"{100 * x:.{nd}f}\\%"


def fmt_int(x) -> str:
    if x is None:
        return NA
    return f"{int(round(x)):,}".replace(",", "{,}")


def fmt_dec(x: float, nd: int) -> str:
    if x is None:
        return NA
    return f"{x:.{nd}f}"


def fmt_hz(x: float) -> str:
    if x is None:
        return NA
    return f"{x:.2f}" if x < 1 else f"{x:.1f}"


class Macros:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []  # (section, name, value)
        self.names: set[str] = set()
        self.section = ""

    def sec(self, title: str) -> None:
        self.section = title

    def add(self, name: str, value: str) -> None:
        full = "Num" + name
        if not re.fullmatch(r"[A-Za-z]+", full):
            raise ValueError(f"macro name not letters-only: {full}")
        if full in self.names:
            raise ValueError(f"duplicate macro {full}")
        self.names.add(full)
        self.items.append((self.section, full, value))

    def add_ci(self, name: str, val: float, lo: float, hi: float, f=fmt_pct) -> None:
        self.add(name, f(val))
        self.add(name + "Lo", f(lo))
        self.add(name + "Hi", f(hi))
        self.add(name + "CI", f"[{f(lo)}, {f(hi)}]")

    def render(self, header: str) -> str:
        lines = [header]
        cur = None
        for sec, name, val in self.items:
            if sec != cur:
                lines.append(f"\n% ---- {sec}")
                cur = sec
            lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- helpers
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def exposure_row(d, event, window, k, R, vis="main"):
    rows = [r for r in d["exposure"] if r["dataset"] == "UAVDT" and r["event"] == event and r["window"] == window
            and r["k"] == k and r["R"] == R and r["vis_def"] == vis and r["group"] == "VEHICLE"]
    if len(rows) != 1:
        raise KeyError((event, window, k, R, len(rows)))
    return rows[0]


EVT = {"E2": "ETwo", "E3": "EThree", "E1": "EOne"}
WIN = {"birth_W120": "Birth", "all_v1": "Vone", "km": "KM"}
RREC = {"1.0": "ROne", "0.8": "RPtEight", "0.5": "RPtFive", "0.95": "RPtNineFive", "0.3": "RPtThree"}
GROUP = {"low": "Low", "mid": "Mid", "high": "High"}


def rname(r) -> str:
    return RREC[f"{float(r):.1f}" if float(r) in (1.0, 0.8, 0.5, 0.3) else f"{float(r):g}"]


def theory_constants() -> dict:
    """Illustration constants of THEORY_28 §3/§4, parsed from code/theory_checks.py (not re-typed)."""
    src = THEORY_PY.read_text(encoding="utf-8")
    m = re.search(r"theta, h, vmax, eps = math\.radians\(([\d.]+)\), ([\d.]+), ([\d.]+), ([\d.]+)", src)
    m2 = re.search(r"kappa, s0, rmax = ([\d.]+), ([\d.]+), ([\d.]+)", src)
    if not (m and m2):
        raise RuntimeError("illustration constants not found in code/theory_checks.py")
    return dict(theta_deg=float(m.group(1)), h=float(m.group(2)), vmax=float(m.group(3)), eps=float(m.group(4)),
                kappa=float(m2.group(1)), s0=float(m2.group(2)), rmax=float(m2.group(3)))


# ----------------------------------------------------------------------------- P0 / P0b
def p0_numbers(M: Macros, d: dict) -> None:
    U = d["data"]["UAVDT"]
    meta = d["meta"]
    M.sec("data and protocol constants (p0_stats.json: data, meta, constants)")
    M.add("UAVDTSeq", fmt_int(U["n_seq"]))
    M.add("UAVDTSeqTrain", fmt_int(U["n_seq_by_split"]["train"]))
    M.add("UAVDTSeqTest", fmt_int(U["n_seq_by_split"]["test"]))
    M.add("UAVDTFrames", fmt_int(U["n_frames_total"]))
    M.add("UAVDTTracks", fmt_int(U["n_tracks"]))
    M.add("UAVDTBoxes", fmt_int(U["n_boxes"]))
    M.add("UAVDTSeqLenMin", fmt_int(U["seq_len_min"]))
    M.add("UAVDTSeqLenMax", fmt_int(U["seq_len_max"]))
    for a, n in U["altitude_counts"].items():
        M.add(f"UAVDTAlt{a.capitalize()}Seq", fmt_int(n))
    M.add("UAVDTFps", fmt_int(d["constants"]["expected"]["UAVDT"]["fps"]))
    vd = d["constants"]["expected"]["VisDrone"]
    M.add("VisDroneClips", fmt_int(vd["clips"]["total"]))
    M.add("VisDroneFrames", fmt_int(vd["frames"]["total"]))
    M.add("WBirth", fmt_int(meta["W_birth"]))
    M.add("BootB", fmt_int(meta["B"]))
    M.add("Seed", fmt_int(meta["seed"]))
    M.add("CIPct", fmt_int(d["constants"]["thresholds"]["ci_pct"]) + "\\%")
    M.add("KGridMin", fmt_int(min(meta["K_GRID"])))
    M.add("KGridMax", fmt_int(max(meta["K_GRID"])))
    M.add("RGridMin", fmt_int(min(meta["R_GRID"])))
    M.add("RGridMax", fmt_int(max(meta["R_GRID"])))
    M.add("SMax", fmt_int(max(meta["K_GRID"]) * max(meta["R_GRID"])))
    for S in sorted({k * R for k in meta["CI_K"] for R in meta["CI_R"]}):
        M.add(f"S{w(S)}", fmt_int(S))  # strides of the CI grid (S = kR)
    for r in meta["R_RECALL"]:
        M.add(f"Val{RREC[str(float(r))]}", f"{r:g}")  # per-look recall levels
    M.add("SizeBinA", fmt_int(d["constants"]["thresholds"]["size_bins"][0]))
    M.add("SizeBinB", fmt_int(d["constants"]["thresholds"]["size_bins"][1]))
    M.add("SizeBinC", fmt_int(d["constants"]["thresholds"]["size_bins"][2]))
    M.add("DShortThr", fmt_int(8))  # D<8 tail threshold, as keyed in p0_stats (share_D_lt_8)

    M.sec("censoring (p0_stats.json: censoring)")
    for key, ev in [("E2|main|VEHICLE", "E2"), ("E3|main|VEHICLE", "E3"), ("E1|main|VEHICLE", "E1")]:
        c = d["censoring"]["UAVDT"][key]
        e = EVT[ev]
        M.add(f"Cens{e}All", fmt_int(c["n_all"]))
        M.add(f"Cens{e}Right", fmt_int(c["n_right_censored"]))
        M.add(f"Cens{e}RightShare", fmt_pct(c["share_right_censored"]))
        M.add(f"Cens{e}Left", fmt_int(c["n_left_censored"]))
        M.add(f"Cens{e}Birth", fmt_int(c["n_birth_window"]))
        M.add(f"Cens{e}BirthRight", fmt_int(c["n_birth_window_right_censored"]))
        M.add(f"Cens{e}BirthMinDobs", fmt_int(c["birth_censored_min_D_obs"]))
        M.add(f"Cens{e}KM", fmt_int(c["n_km"]))
        if "birth_D_lt_8_n" in c:
            M.add(f"Cens{e}AllDShortN", fmt_int(c["all_D_lt_8_n"]))
            M.add(f"Cens{e}AllDShortBorder", fmt_pct(c["all_D_lt_8_share_border"]))
            M.add(f"Cens{e}BirthDShortN", fmt_int(c["birth_D_lt_8_n"]))
            M.add(f"Cens{e}BirthDShortBorder", fmt_pct(c["birth_D_lt_8_share_border"]))

    M.sec("exposure: share_short and miss_r (p0_stats.json: exposure; CI = sequence bootstrap)")
    for ev in ("E2", "E3"):
        for k in meta["CI_K"]:
            for R in meta["CI_R"]:
                S = k * R
                for win in ("birth_W120", "all_v1", "km"):
                    r = exposure_row(d, ev, win, k, R)
                    base = f"{EVT[ev]}S{w(S)}{WIN[win]}"
                    M.add_ci(f"ShareShort{base}", r["share_short"], r["share_short_lo"], r["share_short_hi"])
                    for rr in ("0.8", "0.5"):
                        key = f"miss_r{rr}"
                        M.add_ci(f"Miss{RREC[rr]}{base}", r[key], r[key + "_lo"], r[key + "_hi"])
    # short aliases for the headline numbers (E3, birth window)
    M.sec("exposure: headline aliases (E3, birth window W)")
    for S, (k, R) in [(50, (5, 10)), (120, (12, 10))]:
        r = exposure_row(d, "E3", "birth_W120", k, R)
        M.add_ci(f"ShareShortEThreeS{w(S)}", r["share_short"], r["share_short_lo"], r["share_short_hi"])
        M.add(f"ShareShortEThreeS{w(S)}KOf", fmt_int(k))
        M.add(f"ShareShortEThreeS{w(S)}ROf", fmt_int(R))

    M.sec("D quantiles, frames (p0_stats.json: D_quantiles, D_quantiles_km)")
    for ev in ("E2", "E3"):
        q = d["D_quantiles"]["UAVDT"][f"{ev}|main|VEHICLE|birth"]
        qk = d["D_quantiles_km"]["UAVDT"][f"{ev}|main|VEHICLE"]
        for p, pn in (("p10", "PTen"), ("p50", "PFifty"), ("p90", "PNinety")):
            # birth-window quantiles above W are lower bounds (p*_exact False) -> prefix with \geq
            geq = "" if q.get(f"{p}_exact", True) else "\\ensuremath{\\geq}"
            M.add(f"D{EVT[ev]}Birth{pn}", f"{geq}{fmt_int(q[p])}")
            M.add(f"D{EVT[ev]}KM{pn}", fmt_int(qk[p]))
        M.add(f"D{EVT[ev]}BirthN", fmt_int(q["n"]))
        M.add(f"D{EVT[ev]}BirthShortShare", fmt_pct(q["share_D_lt_8"]))
        M.add(f"D{EVT[ev]}KMShortShare", fmt_pct(qk["share_D_lt_8"]))

    M.sec("density M(t), E0 activity (p0_stats.json: E0, E3_arrival, D_quantiles M_at_birth)")
    e0 = d["E0"]["UAVDT"]["main"]
    M.add("MMedian", fmt_int(e0["M_median"]))
    M.add("MPTen", fmt_int(e0["M_p10"]))
    M.add("MPNinety", fmt_int(e0["M_p90"]))
    M.add("MMax", fmt_int(e0["M_max"]))
    M.add("MShareOneFive", fmt_pct(e0["share_frames_M_1_5"]))
    M.add("MShareSixTwenty", fmt_pct(e0["share_frames_M_6_20"]))
    M.add("MShareGtTwenty", fmt_pct(e0["share_frames_M_gt_20"]))
    M.add("RhoEvMedian", fmt_pct(e0["rho_ev_median"]))
    M.add("RhoEvPooled", fmt_pct(e0["rho_ev_pooled"]))
    M.add("RhoEvMin", fmt_pct(e0["rho_ev_min"]))
    mb = d["D_quantiles"]["UAVDT"]["E3|main|VEHICLE|M_at_birth"]
    M.add("MAtBirthMedian", fmt_int(mb["median"]))
    M.add("EThreeArrivalPerHundred", fmt_dec(d["E3_arrival"]["UAVDT"]["main"]["pooled_per_100f"], 2))

    p = d["p0b"]
    M.sec("ROI sensitivity (p0_stats.json: p0b.roi)")
    for mk, mname in [("m0.00", "MZero"), ("m0.05", "MFive"), ("m0.10", "MTen")]:
        x = p["roi"]["UAVDT"][mk]
        b = f"Roi{mname}"
        M.add(b + "Enter", fmt_int(x["n_enter"]))
        M.add(b + "NotEnter", fmt_int(x["n_not_enter"]))
        M.add(b + "Birth", fmt_int(x["n_birth"]))
        M.add(b + "DShortN", fmt_int(x["D_lt_8_n"]))
        M.add(b + "DShortShare", fmt_pct(x["D_lt_8_share"]))
        M.add(b + "DShortBorderN", fmt_int(x["D_lt_8_border_n"]))
        M.add(b + "DShortInteriorN", fmt_int(x["D_lt_8_interior_n"]))
        M.add(b + "DShortRemaining", fmt_pct(x["D_lt_8_remaining_vs_m0"]))
        M.add(b + "DMedian", fmt_int(x["D_birth"]["p50"]))
        for ck, cell in x["cells"].items():
            S = cell["S"]
            M.add_ci(f"{b}ShareShortS{w(S)}", cell["share_short"], *cell["share_short_ci"])
    M.add("RoiMarginFive", fmt_int(5) + "\\%")
    M.add("RoiMarginTen", fmt_int(10) + "\\%")

    M.sec("iso-miss S*(eps) (p0_stats.json: p0b.isomiss; S in 1..S_scan_max)")
    M.add("IsoSScanMax", fmt_int(p["meta"]["S_scan_max"]))
    for key, x in p["isomiss"]["UAVDT"].items():
        mk, rk, est = key.split("|")
        mname = {"m0.00": "MZero", "m0.05": "MFive"}[mk]
        rname = RREC[rk[1:]]
        ename = {"birth": "", "km": "KM"}[est]
        for ek, v in x.items():
            if not ek.startswith("eps"):
                continue
            eps = float(ek[3:])
            b = f"Iso{mname}{rname}{ename}Eps{wpct(eps)}"
            M.add_ci(b + "S", v["S_star"], *v["S_star_ci"], f=fmt_int)
            M.add_ci(b + "Hz", v["hz_star"], *v["hz_star_ci"], f=fmt_hz)
            M.add_ci(b + "A", v["a_star"], *v["a_star_ci"], f=lambda t: fmt_dec(t, 3))
    for e in p["meta"]["eps"]:
        M.add(f"Eps{wpct(e)}", fmt_pct(e, 0))

    M.sec("onset latency: largest S with P(L<=Lmax) >= p (p0_stats.json: p0b.latency)")
    for key, x in p["latency"]["UAVDT"].items():
        mk, rk, lk = key.split("|")
        mname = {"m0.00": "MZero", "m0.05": "MFive"}[mk]
        rname = RREC[rk[1:]]
        L = int(lk[1:])
        b = f"Lat{mname}{rname}L{w(L)}"
        for pk, v in x["S_needed"].items():
            pname = {"p0.95": "PNinetyFive", "p0.99": "PNinetyNine"}[pk]
            M.add_ci(b + pname + "S", v["S_star"], *v["S_star_ci"], f=fmt_int)
            M.add(b + pname + "Hz", fmt_hz(v["hz_star"]) if v["S_star"] > 0 else "--")
        if mk == "m0.00" and rk == "r1.0":
            M.add(f"LmaxL{w(L)}Sec", fmt_dec(x["Lmax_s"], 2))
            M.add(f"LmaxL{w(L)}", fmt_int(L))
    for pk in p["meta"]["p_lat"]:
        M.add({0.95: "PLatNinetyFive", 0.99: "PLatNinetyNine"}[pk], fmt_pct(pk, 0))

    M.sec("onset-window density rho_onset(w) and onset rate lambda (p0_stats.json: p0b.rho_onset)")
    for key, x in p["rho_onset"]["UAVDT"].items():
        mk, wk = key.split("|")
        mname = {"m0.00": "MZero", "m0.05": "MFive"}[mk]
        b = f"Rho{mname}W{w(int(wk[1:]))}"
        M.add_ci(b, x["pooled"], *x["pooled_ci"])
        M.add(b + "SeqMedian", fmt_pct(x["seq_median"]))
        M.add(b + "SeqPTen", fmt_pct(x["seq_p10"]))
        M.add(b + "SeqPNinety", fmt_pct(x["seq_p90"]))
    for mk, mname in [("m0.00", "MZero"), ("m0.05", "MFive")]:
        lam = p["rho_onset"]["UAVDT"][f"{mk}|w3"]["lambda_per_frame"]
        M.add(f"Lambda{mname}", fmt_dec(lam, 4))
        M.add(f"Lambda{mname}Pct", fmt_pct(lam, 2))
    for wv in p["meta"]["w_onset"]:
        M.add(f"WOnset{w(wv)}", fmt_int(wv))

    M.sec("CP0 decision (p0_stats.json: decision, constants.thresholds)")
    dec = d["decision"]["UAVDT"]
    th = d["constants"]["thresholds"]
    M.add("DecisionBThr", fmt_pct(th["B_share_short"], 0))
    M.add("DecisionCThr", fmt_pct(th["C_share_short"], 0))
    M.add("VoneShareShortEThreeSFifty", fmt_pct(dec["v1_share_short_E3_k5_R10"]))
    M.add("VoneShareShortEThreeSOneHundredTwenty", fmt_pct(dec["v1_share_short_E3_k12_R10"]))


# ----------------------------------------------------------------------------- theory
def theory_numbers(M: Macros, t: dict) -> None:
    checks = t["checks"]
    M.sec("theory checks (theory_checks.json: checks)")
    M.add("ThChecks", fmt_int(len(checks)))
    M.add("ThChecksPass", fmt_int(sum(c["passed"] for c in checks)))
    M.add("ThChecksFail", fmt_int(sum(not c["passed"] for c in checks)))
    by = {c["section"]: c for c in checks}
    l1 = by["§2 Lemma 1"]
    M.add("ThLemOneComparisons", fmt_int(l1["n_comparisons"]))
    M.add("ThLemOneMaxZ", fmt_dec(l1["max_z"], 2))
    M.add("ThLemOneChiP", fmt_dec(l1["chi2_p"], 2))
    M.add("ThLemOneIiiPairs", fmt_int(by["§2 Lemma 1(ii)"]["n"]))
    M.add("ThCorOneSeqs", fmt_int(by["§2 Cor. 1"]["n"]))
    lat = by["§3 Cor. P1-lat"]
    M.add("ThLatComparisons", fmt_int(lat["n_comparisons"]))
    M.add("ThLatMaxZ", fmt_dec(lat["max_z"], 3))
    M.add("ThLatExceed", fmt_int(lat["n_exceed_3sigma"]))
    M.add("ThLatExceedExpected", fmt_dec(lat["expected_exceed_3sigma_if_true"], 2))
    M.add("ThLatChiP", fmt_dec(lat["chi2_p"], 4))
    g1 = by["§5 Prop. G1"]
    M.add("ThGOneMaxZ", fmt_dec(g1["max_z"], 2))
    g2 = by["§5 Prop. G2"]
    M.add("ThGTwoSettings", fmt_int(g2["n_comparisons"]))
    M.add("ThGTwoMaxZ", fmt_dec(g2["max_z"], 2))
    g3 = by["§5 Thm G3"]
    M.add("ThGThreeConfigs", fmt_int(g3["n"]))
    M.add("ThGThreeMaxErr", f"{g3['max_err']:.1e}".replace("e-", "\\times10^{-").replace("e+", "\\times10^{") + "}")
    M.add("ThGFourConfigs", fmt_int(by["§5 Cor. G4"]["n"]))
    full = by["§5 kiểm toàn chuỗi"]["detail"]
    M.add("ThFullWinPred", fmt_dec(full["win"]["delta_pred_sparse"], 3))
    M.add("ThFullWinSim", fmt_dec(full["win"]["delta_mc"], 3))
    M.add("ThFullWinSE", fmt_dec(full["win"]["delta_se"], 3))
    M.add("ThFullLosePred", fmt_dec(full["lose"]["delta_pred_sparse"], 5))
    M.add("ThFullLoseSim", fmt_dec(full["lose"]["delta_mc"], 5))
    M.add("ThFullLoseSE", fmt_dec(full["lose"]["delta_se"], 4))
    M.add("ThGFiveMaxZ", fmt_dec(by["§5 Cor. G5"]["max_z"], 2))
    M.add("ThPThreeSettings", fmt_int(by["§6 Thm P3(i)"]["n"]))
    g2p = next((c for c in checks if "Remark G2" in c["section"]), None)
    if g2p:  # THEORY_28 v1.1, Remark G2'
        M.add("ThGTwoPrimeComparisons", fmt_int(g2p["n_comparisons"]))
        M.add("ThGTwoPrimeMaxZ", fmt_dec(g2p["max_z"], 2))
        M.add("ThGTwoPrimeChiP", fmt_dec(g2p["chi2_p"], 2))
        M.add("ThGTwoPrimeCfg", fmt_int(g2p["n_cfg"]))
        M.add("ThGTwoPrimeCfgInterior", fmt_int(g2p["n_cfg_with_interior_peak"]))
        M.add("ThGTwoPrimeRdaggerErr", f"{g2p['r_dagger_closed_max_err']:.0e}".replace("e-0", "e-").replace("e-", "\\times10^{-") + "}")

    M.sec("P1 example (theory_checks.json: checks[§3 Thm P1]; constants from code/theory_checks.py)")
    C = theory_constants()
    M.add("POneH", fmt_int(C["h"]))
    M.add("POneTheta", fmt_int(C["theta_deg"]))
    M.add("POneVmax", fmt_int(C["vmax"]))
    M.add("POneEps", fmt_pct(C["eps"], 0))
    p1 = by["§3 Thm P1"]
    M.add("POneL", fmt_dec(p1["L_m"], 1))
    det = p1["detail"]
    Fs = sorted({int(k.split("|F")[1]) for k in det})
    M.add("POneFMin", fmt_int(min(Fs)))
    M.add("POneFMax", fmt_int(max(Fs)))
    M.add("POneSettings", fmt_int(len(det)))
    M.add("POneMissMax", fmt_pct(max(v["miss_mc"] for v in det.values()), 2))
    M.add("POneAllNmin", "all" if all(v["N_min"] == v["n"] for v in det.values()) else "not all")
    for rk in sorted({k.split("|")[0] for k in det}):
        v = det[f"{rk}|F{Fs[0]}"]
        rn = RREC[rk[1:]]
        M.add(f"POne{rn}N", fmt_int(v["n"]))
        M.add(f"POne{rn}FdetReq", fmt_dec(v["f_det_req"], 3))

    M.sec("P2 example (theory_checks.json: P2_example; constants from code/theory_checks.py)")
    M.add("PTwoKappa", fmt_int(C["kappa"]))
    M.add("PTwoSZero", fmt_int(C["s0"]))
    M.add("PTwoRmax", fmt_dec(C["rmax"], 1))
    M.add("PTwoPxAtFifty", fmt_int(C["kappa"] / 50))  # object size in px at h = 50 m
    e = t["P2_example"]
    for bk, bn in [("loglogistic_beta3.0", "BetaThree"), ("loglogistic_beta1.5", "BetaOnePtFive")]:
        M.add(f"PTwo{bn}Hstar", fmt_dec(e[bk]["foc_h"], 1))
        M.add(f"PTwo{bn}Happrox", fmt_dec(e[bk]["approx_h_small_r"], 1))
    M.add("PTwoBetaThree", fmt_dec(e["loglogistic_beta3.0"]["beta"], 0))
    M.add("PTwoBetaOnePtFive", fmt_dec(e["loglogistic_beta1.5"]["beta"], 1))
    M.add("PTwoNoInteriorBetaA", fmt_dec(e["loglogistic_beta0.8"]["beta"], 1))
    M.add("PTwoNoInteriorBetaB", fmt_dec(e["loglogistic_beta1.0"]["beta"], 1))
    M.add("PTwoHGridMax", fmt_int(e["loglogistic_beta0.8"]["argmax_h"]))
    M.add("PTwoFloorR", fmt_dec(e["logistic_floor"]["r_floor"], 3))
    M.add("PTwoFloorLocalH", fmt_dec(e["logistic_floor"]["local_max_h"], 1))
    M.add("PTwoIntHstar", fmt_dec(e["integer_n"]["argmin_h"], 1))
    M.add("PTwoIntN", fmt_int(e["integer_n"]["n_at_min"]))

    M.sec("P3 illustration: exact M* and bound M_nec (theory_checks.json: P3_illustration.M_star_table)")
    P3 = t["P3_illustration"]
    M.add("PThreeMMedian", fmt_int(P3["M_median"]))
    for row in P3["M_star_table"]:
        tag = f"R{'Inf' if row['R'] == 'inf' else w(int(row['R']))}Qb{wpct(row['q_b'])}"
        b = f"MStarL{w(row['Lmax'])}Qo{wpct(row['q_o'])}{tag}"
        M.add(b, fmt_int(row["M_star"]))
        M.add(b + "Nec", fmt_dec(row["M_nec"], 1))
    M.sec("P3 illustration: win-set interval (theory_checks.json: P3_illustration.q_in_star_table)")
    for row in P3["q_in_star_table"]:
        b = f"QinL{w(row['Lmax'])}R{'Inf' if row['R'] == 'inf' else w(int(row['R']))}Qout{wpct(row['q_out'])}{RREC[str(row['r'])]}"
        M.add(b + "Star", fmt_dec(row["q_in_star"], 3))
        M.add(b + "Up", fmt_dec(row["q_in_upper"], 3))

    M.sec("iso-KPI illustration: M_iso (theory_checks.json: isoKPI_illustration.table)")
    iso = t["isoKPI_illustration"]
    M.add("IsoLambda", fmt_dec(iso["lambda_per_frame"], 4))
    seen_a = set()
    for row in iso["table"]:
        b = f"MIsoL{w(row['Lmax'])}Eps{wpct(row['eps'])}C{wpct(row['c'])}Qb{wpct(row['q_b'])}Qo{wpct(row['q_o'])}"
        M.add(b, fmt_dec(row["M_iso"], 1))
        ka = (row["Lmax"], row["eps"])
        if ka not in seen_a:
            seen_a.add(ka)
            M.add(f"APeriodicL{w(row['Lmax'])}Eps{wpct(row['eps'])}", fmt_dec(row["a_periodic"], 3))


# ----------------------------------------------------------------------------- simulation (optional)
def sim_numbers(M: Macros, path: Path = SIM) -> int:
    """Macros from results/p2/sim/summary.json (simulated-detector replay). Emits nothing if absent."""
    if not path.exists():
        return 0
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[sim_numbers] cannot read {path}: {e!r} -- skipped")
        return 0
    n0 = len(M.items)
    M.sec("simulated-detector replay (results/p2/sim/summary.json)")
    tr = s.get("theory_repro")
    if tr:
        M.add("SimReproN", fmt_int(tr["n"]))
        M.add("SimReproOk", fmt_int(tr["n_ok"]))
        M.add("SimReproMaxDev", fmt_dec(tr["max_abs_dev"], 4))

    def kname(k: str) -> str:
        out = []
        for part in k.split("|"):
            if part.startswith("r"):
                out.append(RREC.get(f"{float(part[1:]):.1f}" if float(part[1:]) in (1.0, 0.8, 0.5) else part[1:], "R" + wdec(float(part[1:]))))
            elif part.startswith("eps"):
                out.append("Eps" + wpct(float(part[3:])))
            elif part == "miss":
                out.append("Miss")
            elif part.isdigit():
                out.append("L" + w(int(part)))
            else:
                out.append(re.sub(r"[^A-Za-z]", "", part.capitalize()))
        return "".join(out)

    for k, v in (s.get("matched_cost_E3") or {}).items():
        b = "SimMatched" + kname(k)
        for f, n in [("n_cfg", "Cfg"), ("n_gate_win", "GateWin"), ("n_periodic_win", "PeriodicWin"), ("n_holm_win", "HolmWin")]:
            if f in v:
                M.add(b + n, fmt_int(v[f]))
    for k, v in (s.get("iso_kpi") or {}).items():
        b = "SimIso" + kname(k)
        for f, n in [("n_cfg", "Cfg"), ("n_meet", "Meet"), ("n_cheaper", "Cheaper")]:
            if f in v:
                M.add(b + n, fmt_int(v[f]))
        if v.get("a_P_iso") is not None:
            M.add(b + "APeriodic", fmt_dec(v["a_P_iso"], 3))
    dm = s.get("density_Mstar")
    if dm:
        M.add("SimDensityN", fmt_int(dm["n"]))
        M.add("SimDensityAgree", fmt_int(dm["n_agree"]))
    return len(M.items) - n0


# ----------------------------------------------------------------------------- P2B (A1-A3), optional files
SIMDIR = ROOT / "results" / "p2" / "sim"
P2B_FILES = ["mstar_exact.json", "mstar_iso.json", "tail_analysis.json", "multilook.json"]
INF_TEX = "\\ensuremath{\\infty}"


def _load(name: str):
    p = SIMDIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[p2b] cannot read {name}: {e!r} -- skipped")
        return None


def kpi_name(k) -> str:
    k = str(k)
    return "Miss" if k == "miss" else "L" + w(int(k))


def fmt_qo(x) -> str:
    if x is None:
        return NA
    if isinstance(x, float) and math.isinf(x):
        return INF_TEX
    return fmt_dec(x, 3)


def qo_degenerate(row) -> bool:
    ci = row.get("qo_star_pred_ci")
    return row.get("qo_star_pred") == 0 and ci is not None and ci[0] == 0 and ci[1] == 0


def p2b_numbers(M: Macros) -> int:
    n0 = len(M.items)
    ex = _load("mstar_exact.json")
    if ex:
        M.sec("A1 timeline-exact M* check (results/p2/sim/mstar_exact.json)")
        M.add("MStarExactTol", fmt_dec(ex["design"]["tol"], 2))
        M.add("MStarExactSeeds", fmt_int(ex["design"]["K_seeds"]))
        M.add("MStarExactQoN", fmt_int(len(ex["design"]["q_o"])))
        for r, v in ex["by_r"].items():
            b = "MStarExact" + rname(r)
            M.add(b + "N", fmt_int(v["n"]))
            M.add(b + "Match", fmt_int(v["n_match"]))
            M.add(b + "MaxDev", fmt_dec(v["max_abs_dev"], 3))
            M.add(b + "ReplayPlus", fmt_int(v["n_replay_plus"]))
            M.add(b + "ReplayMinus", fmt_int(v["n_replay_minus"]))
            M.add(b + "PredPlus", fmt_int(v["n_sign_pred_plus"]))
        M.add("MStarExactN", fmt_int(ex["total"]["n"]))
        M.add("MStarExactMatch", fmt_int(ex["total"]["n_match"]))
        M.add("MStarExactMatchShare", fmt_pct(ex["total"]["n_match"] / ex["total"]["n"]))
        grp = {}
        for c in ex["cells"]:
            grp.setdefault(c["group"], (c["M_median"], c["n_seq"]))
        for g, (mm, ns) in grp.items():
            M.add(f"Group{GROUP[g]}MMedian", fmt_int(mm))
            M.add(f"Group{GROUP[g]}Seq", fmt_int(ns))

    iso = _load("mstar_iso.json")
    if iso:
        M.sec("A1 iso-KPI boundary q_o* (results/p2/sim/mstar_iso.json); inf = gate cheaper on the whole q_o grid")
        M.add("QoStarN", fmt_int(iso["n"]))
        M.add("QoStarDefined", fmt_int(iso["n_defined"]))
        M.add("QoStarInside", fmt_int(iso["n_inside"]))
        M.add("QoStarSeeds", fmt_int(iso["K_seeds"]))
        M.add("QoStarGridMin", fmt_dec(min(iso["q_o"]), 3))
        M.add("QoStarGridMax", fmt_dec(max(iso["q_o"]), 2))
        for r, v in iso["by_r"].items():
            M.add(f"QoStar{rname(r)}Defined", fmt_int(v["n_defined"]))
            M.add(f"QoStar{rname(r)}Inside", fmt_int(v["n_inside"]))
        rows = iso["rows"]
        n_inf = sum(1 for x in rows if isinstance(x["qo_star_pred"], float) and math.isinf(x["qo_star_pred"]))
        n_deg = sum(1 for x in rows if qo_degenerate(x))
        M.add("QoStarInf", fmt_int(n_inf))
        M.add("QoStarDegenerate", fmt_int(n_deg))
        M.add("QoStarPlotted", fmt_int(len(rows) - n_inf - n_deg))
        for x in rows:
            b = f"QoStar{rname(x['r'])}{GROUP[x['group']]}L{w(x['Lmax'])}Eps{wpct(x['eps'])}"
            M.add(b + "Pred", fmt_qo(x["qo_star_pred"]))
            ci = x.get("qo_star_pred_ci")
            M.add(b + "PredCI", f"[{fmt_qo(ci[0])}, {fmt_qo(ci[1])}]" if ci else NA)
            M.add(b + "Replay", fmt_qo(x["qo_star_replay"]))
            ins = x.get("replay_inside_pred_ci")
            M.add(b + "Inside", NA if ins is None else ("yes" if ins else "no"))
            M.add(b + "APeriodic", fmt_dec(x["a_P_pred"], 3))

    tail = _load("tail_analysis.json")
    if tail:
        M.sec("A2 tail analysis of a_P(eps) (results/p2/sim/tail_analysis.json)")
        M.add("TailNTrain", fmt_int(tail["n_events_train"]))
        M.add("TailNAll", fmt_int(tail["n_events_all"]))
        M.add("TailShareDOneTrain", fmt_pct(tail["share_D_1_train"]))
        M.add("TailShareDLeTwoTrain", fmt_pct(tail["share_D_le_2_train"]))
        M.add("TailShareDLeTwoAll", fmt_pct(tail["share_D_le_2_all"]))
        M.add("TailNDLeTwoTrain", fmt_int(tail["n_D_le_2_train"]))
        M.add("TailShareBorderDLeTwoTrain", fmt_pct(tail["share_border_among_D_le_2_train"]))
        for x in tail["rows"]:
            b = f"Tail{rname(x['r'])}{kpi_name(x['kpi'])}Eps{wpct(x['eps'])}"
            M.add(b + "SIntTrain", fmt_int(x["S_int_train"]))
            M.add(b + "AIntTrain", fmt_dec(x["a_int_train"], 3))
            M.add(b + "SIntAll", fmt_int(x["S_int_all"]))
            M.add(b + "AIntAll", fmt_dec(x["a_int_all"], 3))
            M.add(b + "ARealTrain", fmt_dec(x["a_real_train"], 3))
            M.add(b + "AEngineMZero", fmt_dec(x["a_engine_m0"], 3))
            M.add(b + "AEngineMFive", fmt_dec(x["a_engine_m5"], 3))
        for x in tail["tail"]:
            b = f"TailDrop{rname(x['r'])}{kpi_name(x['kpi'])}Eps{wpct(x['eps'])}Drop{wpct(x['drop_shortest'])}"
            M.add(b + "DMin", fmt_int(x["D_min_kept"]))
            M.add(b + "AReal", fmt_dec(x["a_real"], 3))
        for dv in sorted({x["drop_shortest"] for x in tail["tail"]}):
            M.add(f"TailDropLevel{wpct(dv)}", fmt_pct(dv, 0))

    ml = _load("multilook.json")
    if ml:
        M.sec("A3 Remark G2' multi-look check (results/p2/sim/multilook.json)")
        M.add("MultiN", fmt_int(ml["n"]))
        M.add("MultiObsIncrease", fmt_pct(ml["share_obs_increase"]))
        M.add("MultiPredIncrease", fmt_pct(ml["share_pred_increase"]))
        M.add("MultiSignMatch", fmt_pct(ml["sign_match"]))
        M.add("MultiNDecisive", fmt_int(ml["n_decisive"]))
        M.add("MultiSignMatchDecisive", fmt_pct(ml["sign_match_decisive"]))
        for k, v in ml["by_pair"].items():
            hi, lo = k.split("->")
            b = f"MultiPair{rname(hi)}{rname(lo)}"
            M.add(b + "N", fmt_int(v["n"]))
            M.add(b + "ObsIncrease", fmt_pct(v["obs_increase"]))
            M.add(b + "PredIncrease", fmt_pct(v["pred_increase"]))
            M.add(b + "SignMatch", fmt_pct(v["sign_match"]))
        for k, v in ml["by_kpi"].items():
            b = f"MultiKpi{kpi_name(k)}"
            M.add(b + "N", fmt_int(v["n"]))
            M.add(b + "ObsIncrease", fmt_pct(v["obs_increase"]))
            M.add(b + "SignMatch", fmt_pct(v["sign_match"]))
        cfg = {(x["q_in"], x["q_out"], x["w"], x["R"]) for x in ml["rows"]}
        M.add("MultiCfg", fmt_int(len(cfg)))
    return len(M.items) - n0


# ----------------------------------------------------------------------------- detectors (Kaggle val, VisDrone-DET)
sys.path.insert(0, str(ROOT / "code"))
from paths import DATASETS, model_dir  # noqa: E402

DETECTORS = {"S": "yolo26s_visdrone_1024", "N": "yolo26n_visdrone_1024"}   # macro letter -> paths.MODELS_DIR/<name>


def detector_numbers(M: Macros) -> list:
    """Val VisDrone2019-DET (model.val, max_det=400, end-to-end) from <model>/val_metrics.json; missing -> skipped."""
    used = []
    for k, name in DETECTORS.items():
        p = model_dir(name) / "val_metrics.json"
        if not p.exists():
            continue
        v = json.loads(p.read_text(encoding="utf-8"))
        b = f"DetYolo{k}"
        M.add(b + "MAPFifty", fmt_dec(v["mAP50"], 3))
        M.add(b + "MAPFiftyNinetyFive", fmt_dec(v["mAP50_95"], 3))
        M.add(b + "Precision", fmt_dec(v["precision"], 3))
        M.add(b + "Recall", fmt_dec(v["recall"], 3))
        used.append(p)
    return used


def src_name(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return "$THS_DATASETS/" + p.relative_to(DATASETS).as_posix()


# ----------------------------------------------------------------------------- main
def main() -> int:
    d = json.loads(P0.read_text(encoding="utf-8"))
    t = json.loads(P1.read_text(encoding="utf-8"))
    M = Macros()
    p0_numbers(M, d)
    theory_numbers(M, t)
    n_sim = sim_numbers(M)
    n_sim += p2b_numbers(M)
    det_src = detector_numbers(M)
    src = [P0, P1, THEORY_PY] + ([SIM] if SIM.exists() else []) + [SIMDIR / f for f in P2B_FILES if (SIMDIR / f).exists()] + det_src
    header = ("% numbers.tex -- GENERATED by code/p2/p2_numbers.py. DO NOT EDIT BY HAND.\n"
              f"% generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
              + "".join(f"% input: {src_name(p)}  sha256[:16]={sha(p)}\n" for p in src)
              + (f"% simulation macros: {n_sim}\n" if SIM.exists() else "% simulation: results/p2/sim/summary.json absent -> no Sim* macros\n")
              + "% usage: \\NumXxx{} in text. Percent = one decimal + \\%.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(M.render(header), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(M.items)} macros (sim: {n_sim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
