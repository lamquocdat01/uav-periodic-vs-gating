"""P2B-A2: giải chênh lệch a_P(ε=1%) giữa A3 (P0b: S*=3 ⇒ a≈0,33, toàn bộ UAVDT) và A7 (engine: a=0,462, TRAIN).

Cùng tập E3 cửa sổ sinh, KPI ∈ {miss, L≤3, L≤10}, r ∈ {1, 0.8}, ε ∈ {1, 2, 5 %}; a_P = 1/S* với S* = stride lớn nhất đạt KPI:
  (i)   công thức Lemma 1 trên F_D, S NGUYÊN (như A3)          — tập: TRAIN, và ALL (TRAIN+TEST, như A3)
  (ii)  công thức Lemma 1, S THỰC (lưới 0,01)                   — TRAIN
  (iii) engine replay (detector giả lập r, S thực, 20 pha)      — TRAIN, m=0
  (iv)  engine replay với ROI m = 5 %                            — TRAIN
Đóng góp của đuôi: công thức (i)/(ii) khi bỏ 1 %, 2 %, 5 % sự kiện ngắn nhất; tỉ lệ D ≤ 2 và phần vào từ mép.
Xuất results/p2/sim/tail_analysis.json. Chạy: python code/p2/replay/p2_tail_analysis.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from theory_checks import lemma_miss  # noqa: E402

from p2_dump_schema import DUMPS  # noqa: E402
from p2_engine import EventSet, Timeline, load_events, periodic_iso  # noqa: E402
from p2_match import hits_for_dump  # noqa: E402
from p2_sim_detector import dump_name  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p2" / "sim" / "tail_analysis.json"
EPS = [0.01, 0.02, 0.05]
KPIS = ["miss", 3, 10]
RS = [1.0, 0.8]
DROP = [0.0, 0.01, 0.02, 0.05]


def kpi_miss(D, S, r, kpi):
    T = D if kpi == "miss" else np.minimum(D, kpi + 1)
    return float(np.mean(lemma_miss(T, S, r)))


def s_star_int(D, r, kpi, eps, S_max=400):
    S = 0
    for s in range(1, S_max + 1):
        if kpi_miss(D, s, r, kpi) <= eps + 1e-12:
            S = s
        else:
            break
    return S


def s_star_real(D, r, kpi, eps, S_max=400.0, step=0.01):
    if kpi_miss(D, 1.0, r, kpi) > eps + 1e-12:
        return None
    grid = np.arange(1.0, S_max, step)
    ok = np.array([kpi_miss(D, s, r, kpi) <= eps + 1e-12 for s in grid])
    bad = np.flatnonzero(~ok)
    return float(grid[bad[0] - 1]) if len(bad) else float(grid[-1])


def a_of(S):
    return None if not S else 1.0 / S


def main():
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    ev_tr = load_events("train")
    ev_all = load_events("all")
    D_tr, D_all = ev_tr["E3"].D.values, ev_all["E3"].D.values
    e3 = ev_tr["E3"]
    out = dict(n_events_train=int(len(D_tr)), n_events_all=int(len(D_all)),
               share_D_le_2_train=float((D_tr <= 2).mean()), share_D_le_2_all=float((D_all <= 2).mean()),
               n_D_le_2_train=int((D_tr <= 2).sum()),
               share_border_among_D_le_2_train=float((e3[e3.D <= 2].e3_type == "border").mean()) if (D_tr <= 2).any() else None,
               share_D_1_train=float((D_tr == 1).mean()), rows=[], tail=[])
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group"])
    rows = rows[(rows.group == "VEHICLE") & (rows.split == "train")]
    tl = Timeline(fr[fr.split == "train"].groupby("seq").frame.max().to_dict())
    for r in RS:
        hits = hits_for_dump(DUMPS / dump_name(r, 0.0), rows)
        es0, es5 = EventSet(ev_tr["E3"], hits, tl), EventSet(ev_tr["E3ROI5"], hits, tl)
        for kpi in KPIS:
            for eps in EPS:
                si_tr, si_all = s_star_int(D_tr, r, kpi, eps), s_star_int(D_all, r, kpi, eps)
                sr_tr = s_star_real(D_tr, r, kpi, eps)
                kind, Lm = ("miss", None) if kpi == "miss" else ("lat", kpi)
                Se0, aE0, _ = periodic_iso(tl, es0, eps, kind, Lm)
                Se5, aE5, _ = periodic_iso(tl, es5, eps, kind, Lm)
                out["rows"].append(dict(r=r, kpi=str(kpi), eps=eps,
                                        S_int_train=si_tr, a_int_train=a_of(si_tr), S_int_all=si_all, a_int_all=a_of(si_all),
                                        S_real_train=sr_tr, a_real_train=a_of(sr_tr),
                                        S_engine_m0=Se0, a_engine_m0=aE0, S_engine_m5=Se5, a_engine_m5=aE5))
                print(out["rows"][-1], flush=True)
                for drop in DROP:
                    Ds = np.sort(D_tr)[int(math.floor(drop * len(D_tr))):]
                    si, sr = s_star_int(Ds, r, kpi, eps), s_star_real(Ds, r, kpi, eps)
                    out["tail"].append(dict(r=r, kpi=str(kpi), eps=eps, drop_shortest=drop, D_min_kept=int(Ds.min()),
                                            S_int=si, a_int=a_of(si), S_real=sr, a_real=a_of(sr)))
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print({k: v for k, v in out.items() if k not in ("rows", "tail")})


if __name__ == "__main__":
    main()
