"""P3-D: tái sinh paper/numbers.tex = p2_numbers (đóng băng, tên macro không đổi) + khối macro TEST (tiền tố NumTest…).

Nguồn TEST: results/p3/test_prepare.json, results/p0/audit_prebirth_test.json, $THS_DATASETS/_derived/28/dumps/<lv>/meta.json (TEST),
results/p3/score_test.json, results/p3/cells_test.csv, results/p3/describe_test.json (nếu có).
Kiểm: không trùng tên với macro cũ; tên chỉ chữ cái (p2_numbers.Macros).
Chạy: python code/p3/p3_numbers.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "p2"))
import p2_numbers as P  # noqa: E402
from paths import DUMPS_DIR  # noqa: E402

R3 = ROOT / "results" / "p3"
SRC = [R3 / "test_prepare.json", ROOT / "results" / "p0" / "audit_prebirth_test.json", R3 / "score_test.json", R3 / "cells_test.csv",
       R3 / "describe_test.json"]
TEST_SEQS_KEY = "test_seqs"


def cw(s):
    """'6-20' → SixToTwenty, '>20' → OverTwenty, '1-5' → OneToFive, tên cue → CamelCase."""
    m = {"6-20": "SixToTwenty", ">20": "OverTwenty", "1-5": "OneToFive", "all": "All"}
    if s in m:
        return m[s]
    return "".join(p.capitalize() for p in s.replace("-", "_").split("_"))


def dump_ms(level, seqs):
    meta = json.loads((DUMPS_DIR / level / "meta.json").read_text(encoding="utf-8"))
    by = meta["t_ms_all_median_by_seq"]
    v = [by[s] for s in seqs if s in by]
    import numpy as np
    return float(np.median(v)), len(v)


def test_macros(M):
    prep = json.loads((R3 / "test_prepare.json").read_text(encoding="utf-8"))
    seqs = [r["seq"] for r in prep["sequences"]]
    M.sec("P3 TEST — data (test_prepare.json; code/p3/p3_prepare_test.py)")
    M.add("TestSeq", P.fmt_int(prep["n_seq"]))
    M.add("TestFramesUsed", P.fmt_int(prep["n_frames_used"]))
    M.add("TestFramesDropped", P.fmt_int(prep["n_frames_dropped"]))
    M.add("TestSeqHovering", P.fmt_int(prep["ego_counts"].get("hovering", 0)))
    M.add("TestSeqMoving", P.fmt_int(prep["ego_counts"].get("moving", 0)))
    M.add("TestEThree", P.fmt_int(prep["n_E3_main"]))
    M.add("TestEThreeBorder", P.fmt_int(prep["n_E3_border"]))
    M.add("TestEThreeInterior", P.fmt_int(prep["n_E3_interior"]))
    M.add("TestEThreeBirthWindow", P.fmt_int(prep["n_E3_birth_window"]))
    for s in prep["strata"]:
        nm = f"TestEThree{cw(s['ego'])}M{cw(s['M_bin'])}"
        M.add(nm, P.fmt_int(s["n_events"]))
        M.add(nm + "Seq", P.fmt_int(s["n_seq"]))
    ap = ROOT / "results" / "p0" / "audit_prebirth_test.json"
    if ap.exists():
        a = json.loads(ap.read_text(encoding="utf-8"))
        M.sec("P3 TEST — pre-onset audit, descriptive (audit_prebirth_test.json)")
        lab = {"partial-entry": "PartialEntry", "visibility-transition": "VisTransition", "annotation-late": "AnnotLate",
               "true-birth": "TrueBirth", "censored-start": "CensoredStart"}
        for typ in ("border", "interior", "all"):
            t = a["table_main"][typ]
            for k, nm in lab.items():
                M.add(f"TestAudit{cw(typ)}{nm}", P.fmt_int(t[k]))
            M.add(f"TestAudit{cw(typ)}N", P.fmt_int(t["n"]))
            M.add(f"TestAudit{cw(typ)}AnnotLateShare", P.fmt_pct(t["share_annotation_late"]))
        M.add("TestAuditPlacebo", P.fmt_pct(a["placebo"]["hit_rate"]))
    M.sec("P3 TEST — detector dumps (meta.json, median over TEST sequences of per-sequence median ms/frame)")
    for lv, nm in (("yolo26s_1024", "Main"), ("yolo26n_640", "Low")):  # Main = yolo26s@1024, Low = yolo26n@640
        ms, n = dump_ms(lv, seqs)
        M.add(f"TestDumpMs{nm}", P.fmt_dec(ms, 1))
    s = json.loads((R3 / "score_test.json").read_text(encoding="utf-8"))
    M.sec("P3 TEST — confirmatory family scored (score_test.json; scorer code/p2/p2_prereg_score.py)")
    M.add("TestFamily", P.fmt_int(s["family_size"]))
    M.add("TestScored", P.fmt_int(s["n_scored"]))
    M.add("TestDecided", P.fmt_int(s["n_decided"]))
    tw = s["three_way"]
    for v, nm in (("correct", "Correct"), ("wrong", "Wrong"), ("inconclusive", "Inconclusive")):
        M.add(f"TestN{nm}", P.fmt_int(round(tw[v] * tw["n"])))
        M.add(f"TestShare{nm}", P.fmt_pct(tw[v]))
        for g, d in s["by_group_counts"].items():
            gn = {"H2": "HTwo", "H3-H5": "HThreeFive", "H8": "HEight"}[g]
            M.add(f"Test{gn}N{nm}", P.fmt_int(d[v]))
    for g, d in s["by_group_counts"].items():
        gn = {"H2": "HTwo", "H3-H5": "HThreeFive", "H8": "HEight"}[g]
        M.add(f"Test{gn}N", P.fmt_int(d["n"]))
    cr = s["criteria"]
    M.add("TestCritPShareInconclusive", P.fmt_pct(cr["P_share_inconclusive"]))
    M.add("TestCritOneShareCorrect", P.fmt_pct(cr["i_share_correct_of_decided"]))
    M.add("TestCritThreeShareWrong", P.fmt_pct(cr["iii_H8_share_wrong_of_decided"]))
    M.add("TestHolmSig", P.fmt_int(s["n_sig_holm"] or 0))
    M.add("TestVerdict", s["verdict_words"])
    pc = s["plus_cells"]
    M.add("TestPlusN", P.fmt_int(pc["n"]))
    for v, nm in (("correct", "Correct"), ("wrong", "Wrong"), ("inconclusive", "Inconclusive")):
        M.add(f"TestPlus{nm}", P.fmt_int(pc[v]))
    M.add("TestSizeRuleFail", P.fmt_int(s["n_size_rule_fail"]))
    dp = R3 / "describe_test.json"
    if dp.exists():
        d = json.loads(dp.read_text(encoding="utf-8"))
        M.sec("P3 TEST — descriptive (describe_test.json; not part of the criteria)")
        for var, g in d.get("sensitivity", {}).items():
            for v, nm in (("correct", "Correct"), ("wrong", "Wrong"), ("inconclusive", "Inconclusive")):
                vn = {"onset_alt": "OnsetAlt", "roi5": "RoiFive", "score005": "ScorePtZeroFive", "score050": "ScorePtFifty"}[var]
                M.add(f"TestSens{vn}{nm}", P.fmt_int(g.get(v, 0)))


def main():
    P.main()
    txt = P.OUT.read_text(encoding="utf-8")
    import re
    old = set(re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", txt))
    M = P.Macros()
    test_macros(M)
    dup = old & M.names
    assert not dup, f"trùng tên macro cũ: {sorted(dup)[:5]}"
    hdr = ("\n% ==== P3 TEST macros -- GENERATED by code/p3/p3_numbers.py (appended after p2_numbers; old names unchanged)\n"
           + "".join(f"% input: {p.relative_to(ROOT).as_posix()}  sha256[:16]={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}\n"
                     for p in SRC if p.exists()))
    block = M.render(hdr.rstrip("\n"))
    P.OUT.write_text(txt + block, encoding="utf-8")
    print(f"appended {len(M.items)} TEST macros; old macros {len(old)}")


if __name__ == "__main__":
    main()
