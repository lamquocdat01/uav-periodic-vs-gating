"""P2B-A3: kiểm Remark G2' (multi-look, r < 1) trên kết quả A7 (agnostic_cells.csv, fp = 0, E3, mọi M, KPI latency).

Với mỗi cấu hình (q_in, q_out, w, R) và cặp (r_hi, r_lo) ∈ {(1, 0.8), (0.8, 0.5), (1, 0.5)}:
  quan sát: sign(Δ_obs(r_lo) − Δ_obs(r_hi)), khoảng tin cậy thô [lo_lo − hi_hi, hi_lo − lo_hi] (bảo thủ, không có bootstrap cặp)
  dự đoán:  sign(Δ_pred(r_lo) − Δ_pred(r_hi)), Δ_pred = mean_e[Lemma1(T_e, 1/a, r) − miss_G(T_e; w, R, q_in, q_bg, r)] (mean-field,
            như p2_prereg_build; a = activation dự đoán G1) — trên F_D TRAIN.
Báo tỉ lệ khớp dấu (mọi ô và riêng ô có khoảng quan sát không chứa 0), và tỉ lệ ô "Δ tăng khi r giảm".
Xuất results/p2/sim/multilook.json. Chạy: python code/p2/replay/p2_multilook_check.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_prereg_build import predict_delta, train_events  # noqa: E402

SIM = ROOT / "results" / "p2" / "sim"
PAIRS = [(1.0, 0.8), (0.8, 0.5), (1.0, 0.5)]


def main():
    info = json.loads((SIM / "theory_repro.json").read_text(encoding="utf-8"))["info"]
    rho = {int(k): v for k, v in info["rho_onset"].items()}
    c = pd.read_csv(SIM / "agnostic_cells.csv")
    c = c[(c.fp == 0.0) & (c.event == "E3") & (c.M_bin == "all") & (c.kpi.astype(str).isin(["3", "5", "10"]))]
    D = train_events().D.values
    key = ["q_in", "q_out", "w", "R", "kpi"]
    rows = []
    for k, g in c.groupby(key):
        g = g.set_index("r")
        q_in, q_out, w, R, kpi = k
        Rn = math.inf if str(R) == "inf" else int(float(R))
        pred = {r: predict_delta(D, int(kpi), int(w), Rn, q_in, q_out, 0.0, rho[int(w)], r)[0] for r in (1.0, 0.8, 0.5)}
        for hi, lo in PAIRS:
            if hi not in g.index or lo not in g.index:
                continue
            d_obs = g.loc[lo, "delta"] - g.loc[hi, "delta"]
            ci = [g.loc[lo, "lo"] - g.loc[hi, "hi"], g.loc[lo, "hi"] - g.loc[hi, "lo"]]
            d_pred = pred[lo] - pred[hi]
            rows.append(dict(q_in=q_in, q_out=q_out, w=int(w), R=str(R), kpi=str(kpi), r_hi=hi, r_lo=lo, d_obs=float(d_obs), ci=ci,
                             d_pred=float(d_pred), obs_increase=bool(d_obs > 0), pred_increase=bool(d_pred > 0),
                             obs_decisive=bool(ci[0] > 0 or ci[1] < 0), sign_match=bool((d_obs > 0) == (d_pred > 0))))
    df = pd.DataFrame(rows)
    dec = df[df.obs_decisive]
    out = dict(n=len(df), share_obs_increase=float(df.obs_increase.mean()), share_pred_increase=float(df.pred_increase.mean()),
               sign_match=float(df.sign_match.mean()), n_decisive=int(len(dec)),
               sign_match_decisive=(float(dec.sign_match.mean()) if len(dec) else None),
               by_pair={f"{h}->{l}": dict(n=int(len(g)), obs_increase=float(g.obs_increase.mean()), pred_increase=float(g.pred_increase.mean()),
                                          sign_match=float(g.sign_match.mean()))
                        for (h, l), g in df.groupby(["r_hi", "r_lo"])},
               by_kpi={k: dict(n=int(len(g)), obs_increase=float(g.obs_increase.mean()), sign_match=float(g.sign_match.mean()))
                       for k, g in df.groupby("kpi")},
               rows=rows)
    (SIM / "multilook.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
