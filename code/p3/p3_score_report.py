"""P3-B2: chấm họ xác nhận TEST bằng scorer ĐÃ ĐÓNG BĂNG (code/p2/p2_prereg_score.score) và sinh báo cáo.

Đầu vào: results/p2/prereg_cells.json (sha256 = FREEZE), results/p3/obs_test.json (p3_replay_test.py), results/p3/replay_test_all.parquet.
Xuất: results/p3/score_test.json, results/p3/cells_test.csv, RESULTS_28.md (chỉ bảng + câu chữ tiêu chí, không diễn giải).
Chạy: python code/p3/p3_score_report.py
"""
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_prereg_score import DELTA_MIN, PASS_SHARE, POWER_MAX_INCONCLUSIVE, group, score  # noqa: E402

CELLS = ROOT / "results" / "p2" / "prereg_cells.json"
CELLS_SHA_FREEZE = "fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e"
OUT = ROOT / "results" / "p3"
VERDICT_WORDS = {"confirmed": "confirmed", "rejected": "refuted", "untestable": "not testable"}
VI = {"correct": "ĐÚNG", "wrong": "SAI", "inconclusive": "CHƯA KẾT LUẬN"}
# câu chữ tiêu chí — nguyên văn PREREG_28_DRAFT.md §4 (P2H-Q6)
CRIT_TEXT = {
    "P": "If more than 50 % of the confirmatory cells are INCONCLUSIVE, the paper's conclusion is \"not testable with these data\" "
         "(neither confirmed nor rejected).",
    "i": "fewer than 80 % of the confirmatory cells that are DECIDED (correct or wrong) are correct",
    "ii": "the observed iso-KPI boundary q_o* (THEORY Thm P3-iii; P2B-A1) lies outside its predicted bootstrap CI in ≥ 2/3 of the density groups",
    "iii": "the sign of H8 (\"Δ(latency) increases as recall decreases\") is wrong in ≥ 50 % of detector pairs "
           "(evaluated on decided H8 cells)",
}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def f4(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and (math.isinf(x) or abs(x) >= 1e8):  # −1e9 = mã hoá −∞ của bootstrap H2 (a_P(ε) = ∞)
        return "+∞" if x > 0 else "−∞"
    return f"{x:+.4f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def main():
    sha = hashlib.sha256(CELLS.read_bytes()).hexdigest()
    assert sha == CELLS_SHA_FREEZE, sha
    cells = json.loads(CELLS.read_text(encoding="utf-8"))["cells"]
    conf = {c["id"]: c for c in cells if c["confirmatory"]}
    ob = json.loads((OUT / "obs_test.json").read_text(encoding="utf-8"))
    res = score(cells, ob["obs"], ob.get("boundary"))
    assert res["family_size"] == 72 and res["n_scored"] == 72 and res["n_missing"] == 0, (res["family_size"], res["n_scored"])
    rep = pd.read_parquet(OUT / "replay_test_all.parquet").set_index("id")
    obs_by = {o["id"]: o for o in ob["obs"]}
    rows = []
    for r in res["rows"]:
        c, x, o = conf[r["id"]], rep.loc[r["id"]], obs_by[r["id"]]
        iso = c["comparison"] == "iso_kpi"
        rows.append(dict(id=r["id"], group=group(c["hypothesis"]), hypothesis=c["hypothesis"], comparison=c["comparison"], kpi=c["kpi"],
                         eps=c.get("eps"), ego=c["ego"], M_bin=c["M_bin"], cue=c["cue"], r_level=c["r_level"], sign_pred=c["sign_pred"],
                         pred_value=(c.get("ddelta_pred") if c["comparison"] == "r_pair" else c.get("delta_pred")),
                         pred_a_G=c.get("a_cost") if iso else None, pred_a_P_iso=c.get("a_P_iso") if iso else None,
                         obs_value=(x["diff"] if iso else x["delta"]), lo=(o.get("diff_lo") if iso else o["lo"]),
                         hi=(o.get("diff_hi") if iso else o["hi"]), obs_sign=r["obs"], gate_cheaper=(x.get("gate_cheaper") if iso else None),
                         miss_G=x.get("miss_G"), a_G=x["a_G"], a_P_matched=x["a_P"], a_P_iso=(x.get("a_P_iso") if iso else None),
                         verdict=r["verdict"], verdict_vi=VI[r["verdict"]], p_boot=r.get("p_boot"), p_holm=r.get("p_holm"),
                         sig_holm=r.get("sig_holm"), n_events_test=int(x["n_events_test"]), n_seq_test=int(x["n_seq_test"]),
                         size_rule_test=bool(x["size_rule_test"]), n_events_train=int(c["n_events_train"]), n_seq_train=int(c["n_seq_train"])))
    df = pd.DataFrame(rows).sort_values(["group", "cue", "ego", "M_bin", "kpi"])
    df.to_csv(OUT / "cells_test.csv", index=False, encoding="utf-8")
    plus = df[df.sign_pred == "+"]
    res.update(cells_sha256=sha, freeze_tag=git("describe", "--tags", "--abbrev=0"), head=git("rev-parse", "HEAD"),
               generated=dt.datetime.now().isoformat(timespec="seconds"), verdict_words=VERDICT_WORDS[res["verdict"]],
               criterion_ii_note=ob.get("boundary_note"), criteria_text=CRIT_TEXT, delta_min=DELTA_MIN,
               n_size_rule_fail=int((~df.size_rule_test).sum()),
               plus_cells=dict(n=int(len(plus)), **{v: int((plus.verdict == v).sum()) for v in ("correct", "wrong", "inconclusive")}),
               by_group_counts={g: {v: int((d.verdict == v).sum()) for v in ("correct", "wrong", "inconclusive")} | {"n": int(len(d))}
                                for g, d in df.groupby("group")})
    (OUT / "score_test.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    write_md(df, res)
    print({k: v for k, v in res.items() if k not in ("rows", "criteria_text")})


def write_md(df, res):
    cr = res["criteria"]
    L = ["# RESULTS_28 — UAVDT TEST, họ xác nhận đóng băng (sinh tự động)", "",
         f"*Sinh bởi `code/p3/p3_score_report.py` — {res['generated']}. PREREG: tag `{res['freeze_tag']}`; prereg_cells.json sha256 "
         f"`{res['cells_sha256']}` (= FREEZE). Quan sát: `code/p3/p3_replay_test.py` → results/p3/obs_test.json. Chấm: `code/p2/p2_prereg_score.py` "
         "(đóng băng). File này không chứa diễn giải ngoài câu chữ tiêu chí.*", "",
         "## Tổng hợp 3 mức", "",
         "| nhóm | n | ĐÚNG | SAI | CHƯA KẾT LUẬN |", "|---|---|---|---|---|"]
    for g, d in res["by_group_counts"].items():
        L.append(f"| {g} | {d['n']} | {d['correct']} | {d['wrong']} | {d['inconclusive']} |")
    tw = res["three_way"]
    L.append(f"| **tổng** | {tw['n']} | {round(tw['correct'] * tw['n'])} | {round(tw['wrong'] * tw['n'])} | {round(tw['inconclusive'] * tw['n'])} |")
    p = res["plus_cells"]
    L += ["", f"- Ô dự đoán \"+\": {p['n']} — ĐÚNG {p['correct']}, SAI {p['wrong']}, CHƯA KẾT LUẬN {p['inconclusive']}.",
          f"- Ô không đạt luật cỡ mẫu trên TEST (≥ 30 sự kiện, ≥ 3 chuỗi) → CHƯA KẾT LUẬN: {res['n_size_rule_fail']}.",
          f"- Holm (chỉ báo cáo): {res['n_sig_holm']} ô có ý nghĩa sau hiệu chỉnh.", "",
          "## Tiêu chí (câu chữ PREREG_28_DRAFT §4)", "",
          "| tiêu chí | câu chữ | giá trị TEST | đạt điều kiện |", "|---|---|---|---|",
          f"| P | {CRIT_TEXT['P']} | CHƯA KẾT LUẬN {pct(cr['P_share_inconclusive'])} | {'có' if cr['P_untestable'] else 'không'} |",
          f"| 1 | {CRIT_TEXT['i']} | ĐÚNG / KẾT LUẬN ĐƯỢC = {pct(cr['i_share_correct_of_decided'])} ({res['n_decided']} ô kết luận được) | "
          f"{'có' if cr['i_reject'] else 'không'} |",
          f"| 2 | {CRIT_TEXT['ii']} | không đánh giá: {res['criterion_ii_note']} | — |",
          f"| 3 | {CRIT_TEXT['iii']} | SAI / H8 kết luận được = {pct(cr['iii_H8_share_wrong_of_decided'])} | {'có' if cr['iii_reject'] else 'không'} |",
          "", f"**Kết luận (theo câu chữ tiêu chí): {res['verdict_words']}.**", "",
          "## Bảng 72 ô", "",
          "Δ̂ = miss_P − miss_G ở matched cost (> 0: gate thắng); H8: ΔΔ̂ = Δ̂(yolo26n_640) − Δ̂(yolo26s_1024); H2 (iso-KPI): a_G − a_P(ε). "
          "CI 95 % bootstrap ghép cặp theo chuỗi, B = 1000, seed 42.", "",
          "| nhóm | KPI | ε | ego | M_bin | cue | dự đoán | giá trị dự đoán | quan sát | CI 95 % | gate rẻ hơn (H2) | kết quả | n sự kiện / chuỗi TEST |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in df.itertuples(index=False):
        eps = "—" if x.eps is None or (isinstance(x.eps, float) and math.isnan(x.eps)) else f"{x.eps:g}"
        gc = "—" if x.gate_cheaper is None or (isinstance(x.gate_cheaper, float) and math.isnan(x.gate_cheaper)) else ("có" if x.gate_cheaper else "không")
        pv = f4(x.pred_value) if x.comparison != "iso_kpi" else f"a_G {x.pred_a_G:.3f} / a_P {x.pred_a_P_iso:.3f}"
        L.append(f"| {x.group} | {x.kpi} | {eps} | {x.ego} | {x.M_bin} | {x.cue} | {x.sign_pred} | {pv} | {f4(x.obs_value)} | "
                 f"[{f4(x.lo)}; {f4(x.hi)}] | {gc} | {x.verdict_vi} | {x.n_events_test} / {x.n_seq_test} |")
    L += ["", "Chi tiết: results/p3/cells_test.csv, results/p3/score_test.json."]
    (ROOT / "RESULTS_28.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
