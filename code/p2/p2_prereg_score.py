"""P2H-Q6 (01-10-2026, thay luật P2B 27/09): chấm PREREG 3 mức ĐÚNG / SAI / CHƯA KẾT LUẬN trên kết quả quan sát
(TEST thật sau freeze; hoặc kênh giả lập để thử).

Đầu vào: prereg_cells.json (p2_prereg_build) + quan sát JSON:
  {"obs": [{"id": ..., "delta": float, "lo": float, "hi": float, "p_boot": float}, ...],     matched cost (Δ̂) / H8 (ΔΔ̂)
            {"id": ..., "gate_cheaper": bool, ["diff_lo": float, "diff_hi": float]}, ...],  iso-KPI; diff = a_G − a_P(ε)
   "boundary": [{"name": ..., "obs": q_o* quan sát, "pred_ci": [lo, hi]}, ...]}                 tiêu chí (ii)
CI = bootstrap ghép cặp theo chuỗi, B = 1000, seed 42 (p2_stats.paired_delta).
Luật chấm từng ô xác nhận (δ_min = DELTA_MIN = 0,005 = MARGIN của builder):
  dự đoán "+"   : ĐÚNG iff lo > 0;        SAI iff hi ≤ 0;        còn lại CHƯA KẾT LUẬN
  dự đoán "<=0" : ĐÚNG iff hi < δ_min;    SAI iff lo > δ_min;    còn lại CHƯA KẾT LUẬN
  H8 (ΔΔ̂)      : cùng luật "+"/"<=0"
  iso-KPI (H2)  : ĐÚNG/SAI theo "gate rẻ hơn ở KPI bằng nhau" (quan sát trùng dự đoán); CHƯA KẾT LUẬN khi CI của (a_G − a_P(ε))
                  chứa 0 (chỉ khi quan sát có diff_lo/diff_hi)
  ô dự đoán "—" (H8 ngoài điều kiện r†) không thuộc họ xác nhận, không chấm.
Tiêu chí (cả họ xác nhận):
  P (công suất) : > 50 % ô họ xác nhận CHƯA KẾT LUẬN → kết luận bài = "chưa kiểm được" (không xác nhận, không bác bỏ)
  (i)   < 80 % ô KẾT LUẬN ĐƯỢC là đúng → bác bỏ
  (ii)  q_o* quan sát nằm ngoài CI dự đoán ở ≥ 2/3 nhóm M → bác bỏ
  (iii) dấu H8 sai ở ≥ 50 % ô H8 kết luận được → bác bỏ
  Thứ tự: P trước; nếu P không chặn thì bác bỏ nếu (i) hoặc (ii) hoặc (iii), ngược lại "xác nhận".
  Holm CHỈ trong họ xác nhận, để BÁO số ô có ý nghĩa — không định nghĩa đúng/sai.
Chạy: python code/p2/p2_prereg_score.py --cells results/p2/prereg_cells.json --obs <obs.json> [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "replay"))
from p2_stats import holm  # noqa: E402

PASS_SHARE = 0.80
POWER_MAX_INCONCLUSIVE = 0.50
DELTA_MIN = 0.005
VERDICTS = ("correct", "wrong", "inconclusive")


def judge(pred, lo, hi, delta_min=DELTA_MIN):
    """3 mức cho matched cost / H8."""
    if pred == "+":
        return "correct" if lo > 0 else "wrong" if hi <= 0 else "inconclusive"
    return "correct" if hi < delta_min else "wrong" if lo > delta_min else "inconclusive"


def judge_iso(pred, o):
    if o.get("diff_lo") is not None and o.get("diff_hi") is not None and o["diff_lo"] <= 0 <= o["diff_hi"]:
        return "inconclusive"
    s_obs = "+" if o["gate_cheaper"] else "<=0"
    return "correct" if s_obs == pred else "wrong"


def group(h):
    return "H2" if h.startswith("H2") else "H8" if h == "H8" else "H3-H5" if h.startswith("H3") or h == "H5" else h


def _shares(res):
    n = len(res)
    return {v: (float(np.mean([r["verdict"] == v for r in res])) if n else None) for v in VERDICTS} | {"n": n}


def score(cells, obs, boundary=None):
    cmap = {c["id"]: c for c in cells if c.get("confirmatory") and c.get("sign_pred") != "—"}
    rows = [o for o in obs if o["id"] in cmap]
    res = []
    for o in rows:
        c = cmap[o["id"]]
        if c["comparison"] == "iso_kpi":
            v = judge_iso(c["sign_pred"], o)
            res.append(dict(id=o["id"], pred=c["sign_pred"], obs="+" if o["gate_cheaper"] else "<=0", verdict=v,
                            diff_lo=o.get("diff_lo"), diff_hi=o.get("diff_hi"), hypothesis=c["hypothesis"], comparison="iso_kpi"))
        else:
            s_obs = "+" if o["lo"] > 0 else "-" if o["hi"] < 0 else "0"
            res.append(dict(id=o["id"], pred=c["sign_pred"], obs=s_obs, delta=o["delta"], lo=o["lo"], hi=o["hi"], p_boot=o.get("p_boot"),
                            verdict=judge(c["sign_pred"], o["lo"], o["hi"]), hypothesis=c["hypothesis"], comparison=c["comparison"]))
        res[-1]["correct"] = res[-1]["verdict"] == "correct"
    withp = [r for r in res if r.get("p_boot") is not None]
    n_sig = None
    if withp:
        adj, rej = holm([r["p_boot"] for r in withp])
        for r, a, j in zip(withp, adj, rej):
            r["p_holm"], r["sig_holm"] = float(a), bool(j)
        n_sig = int(sum(rej))
    n = len(res)
    dec = [r for r in res if r["verdict"] != "inconclusive"]
    share_inc = float(np.mean([r["verdict"] == "inconclusive" for r in res])) if n else None
    share_ok = float(np.mean([r["verdict"] == "correct" for r in dec])) if dec else None
    h8 = [r for r in dec if r["hypothesis"] == "H8"]
    h8_wrong = float(np.mean([r["verdict"] == "wrong" for r in h8])) if h8 else None
    crit = dict(P_share_inconclusive=share_inc, P_untestable=bool(share_inc is not None and share_inc > POWER_MAX_INCONCLUSIVE),
                i_share_correct_of_decided=share_ok, i_reject=bool(share_ok is not None and share_ok < PASS_SHARE),
                iii_H8_share_wrong_of_decided=h8_wrong, iii_reject=bool(h8_wrong is not None and h8_wrong >= 0.5))
    if boundary:
        b = [dict(x, inside=bool(x["pred_ci"] is not None and x["pred_ci"][0] <= x["obs"] <= x["pred_ci"][1])) for x in boundary]
        out_share = float(np.mean([not x["inside"] for x in b]))
        crit.update(ii_n_groups=len(b), ii_share_outside=out_share, ii_reject=bool(out_share >= 2 / 3), ii_rows=b)
    rejected = bool(crit["i_reject"] or crit.get("ii_reject", False) or crit["iii_reject"])
    verdict = "untestable" if crit["P_untestable"] else ("rejected" if rejected else "confirmed")
    out = dict(family_size=len(cmap), n_scored=n, n_missing=len(cmap) - n, n_decided=len(dec), share_correct=share_ok,
               share_inconclusive=share_inc, n_sig_holm=n_sig, rejected=rejected, verdict=verdict, criteria=crit,
               three_way=_shares(res), three_way_by_group={g: _shares([r for r in res if group(r["hypothesis"]) == g])
                                                         for g in sorted({group(r["hypothesis"]) for r in res})},
               by_hypothesis={h: {v: int(sum(1 for r in res if r["hypothesis"] == h and r["verdict"] == v)) for v in VERDICTS}
                              for h in sorted({r["hypothesis"] for r in res})},
               rows=res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cells = json.loads(Path(a.cells).read_text(encoding="utf-8"))["cells"]
    ob = json.loads(Path(a.obs).read_text(encoding="utf-8"))
    obs, bnd = (ob, None) if isinstance(ob, list) else (ob["obs"], ob.get("boundary"))
    out = score(cells, obs, bnd)
    Path(a.out or Path(a.obs).with_name("prereg_score.json")).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
