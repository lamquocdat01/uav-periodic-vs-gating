"""P2B-A1: kiểm M* ĐÚNG CÁCH (thay §4 cũ của P2_SIM_REPORT): dự đoán bằng công thức TIMELINE-EXACT cho mọi r.

Cue phụ thuộc mật độ: q_in = 1 trong cửa sổ khởi phát [onset, onset+w) (w = Lmax), ngoài cửa sổ q_out,t = 1 − (1 − q_o)^{M_t}
(M_t = số xe visible thật ở khung t), R = ∞. UAVDT TRAIN chia 3 nhóm chuỗi theo M trung vị (tertile). KPI latency Lmax ∈ {3, 5}.
  Dự đoán (không dùng replay):   a_pred = mean_t P(chạy_t) = mean_t q_t   (q_t = 1 nếu I_t, ngược lại q_out,t)
                                 miss_P,e = Lemma1(min(D_e, Lmax+1), S = 1/a_pred, r)
                                 miss_G,e = Π_{t ∈ cửa sổ KPI} (1 − r q_t)          (Prop. G2 timeline-exact, R = ∞)
                                 Δ_pred = mean_e (miss_P,e − miss_G,e)
  Replay: detector giả lập r, cue gieo K = 5 lần (trung bình theo seed), periodic matched cost (bisection).
  Khớp: Δ_pred ∈ CI 95% (bootstrap theo chuỗi) của Δ_replay, hoặc cùng dấu và |lệch| < 0.01.
  q_o* = điểm Δ đổi dấu (+ → ≤ 0) trên lưới q_o, nội suy tuyến tính; CI bootstrap theo chuỗi — cho cả replay và dự đoán.
A-sparse (THEORY §5.1) chỉ còn là cận, ghi kèm để so.
Xuất results/p2/sim/mstar_exact.json. Chạy: python code/p2/replay/p2_mstar_check.py
"""
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from theory_checks import can_win, lemma_miss, q_out_eff  # noqa: E402

from p2_dump_schema import DUMPS  # noqa: E402
from p2_engine import (EventSet, Timeline, activation, gate_or_refresh, kpis, load_events, matched_periodic_S,  # noqa: E402
                       onset_frames, onset_mask, periodic)
from p2_match import hits_for_dump  # noqa: E402
from p2_sim_detector import dump_name  # noqa: E402
from p2_stats import boot_weights, crossing, paired_delta  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p2" / "sim" / "mstar_exact.json"
RS = [1.0, 0.8, 0.5]
Q_O = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
LMAX = [3, 5]
K_SEEDS = 5
TOL = 0.01
SEED = 42


def seed_of(*p):
    return SEED + zlib.crc32("|".join(map(str, p)).encode()) % (2 ** 31)


def pred_event_vectors(es, qt, Lm, r):
    """(miss_G,e) timeline-exact, R = ∞, cửa sổ KPI = min(D, Lmax+1)."""
    tl = es.tl
    T = np.minimum(es.ev.D.values, Lm + 1)
    out = np.empty(es.n)
    for i, (sidx, st, Te) in enumerate(zip(es.seq_idx, es.ev.start.values, T)):
        g0 = tl.off[sidx] + st - 1
        out[i] = np.prod(1 - r * qt[g0: g0 + Te])
    return out, T


def main():
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    fr = fr[fr.split == "train"]
    nfr = fr.groupby("seq").frame.max().to_dict()
    tl_all = Timeline(nfr)
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group"])
    rows = rows[(rows.group == "VEHICLE") & (rows.split == "train")]
    ev = load_events("train")
    ons = onset_frames("train")
    seq_M = fr.groupby("seq").M.median()
    terc = pd.qcut(seq_M.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    cells, curves = [], []
    for r in RS:
        hits = hits_for_dump(DUMPS / dump_name(r, 0.0), rows)
        for grp in ("low", "mid", "high"):
            seqs = [s for s in tl_all.seqs if terc[s] == grp]
            tl = Timeline({s: nfr[s] for s in seqs})
            Mg = np.zeros(tl.G)
            for s in seqs:
                g = fr[fr.seq == s].sort_values("frame")
                Mg[tl.g(s, g.frame.values)] = g.M.values
            es = EventSet(ev["E3"][ev["E3"].seq.isin(seqs)], hits, tl)
            W = boot_weights(len(seqs))
            M_med = float(np.median(Mg))
            for Lm in LMAX:
                w = Lm
                im = onset_mask(tl, ons[ons.seq.isin(seqs)], w)
                rho = float(im.mean())
                d_rep_bs, d_pred_bs, d_rep_pt, d_pred_pt = [], [], [], []
                for q_o in Q_O:
                    qo_t = q_out_eff(0.0, q_o, Mg)
                    qt = np.where(im, 1.0, qo_t)
                    # ---- dự đoán
                    a_pred = float(qt.mean())
                    mG_pred, T = pred_event_vectors(es, qt, Lm, r)
                    mP_pred = lemma_miss(T, 1.0 / a_pred, r)
                    dp = paired_delta(mP_pred, mG_pred, es.seq_idx, len(seqs), W)
                    # ---- replay (K seed cue)
                    mG_rep, mP_rep, aG = np.zeros(es.n), np.zeros(es.n), []
                    for k in range(K_SEEDS):
                        u = np.random.default_rng(seed_of(r, grp, Lm, q_o, k)).random(tl.G)
                        runG = gate_or_refresh(u < qt, tl, math.inf)
                        aG.append(float(activation(runG).mean()))
                        mG_rep += 1 - kpis(es.first_latency(runG), (Lm,))[f"L{Lm}"]
                    mG_rep /= K_SEEDS
                    S_m, aP = matched_periodic_S(tl, float(np.mean(aG)))
                    mP_rep = 1 - kpis(es.first_latency(periodic(tl, S_m)), (Lm,))[f"L{Lm}"]
                    dr = paired_delta(mP_rep, mG_rep, es.seq_idx, len(seqs), W)
                    asp = can_win(Lm + 1, w, math.inf, rho, float(q_out_eff(0.0, q_o, M_med)), r)
                    match = bool(dr["lo"] <= dp["delta"] <= dr["hi"] or
                                 (np.sign(dp["delta"]) == np.sign(dr["delta"]) and abs(dp["delta"] - dr["delta"]) < TOL))
                    cells.append(dict(r=r, group=grp, n_seq=len(seqs), M_median=M_med, Lmax=Lm, w=w, rho=rho, q_o=q_o,
                                      a_pred=a_pred, a_replay=float(np.mean(aG)), a_P=aP, n_events=es.n,
                                      delta_pred=dp["delta"], delta_pred_lo=dp["lo"], delta_pred_hi=dp["hi"],
                                      delta_replay=dr["delta"], lo=dr["lo"], hi=dr["hi"], sign_replay=dr["sign"],
                                      sign_pred=("+" if dp["delta"] > 0 else "<=0"), match=match,
                                      asparse_can_win_at_M_median=bool(asp[0]), asparse_best_delta=asp[2]))
                    # vector bootstrap cho q_o*
                    s_rep = np.bincount(es.seq_idx, weights=mP_rep - mG_rep, minlength=len(seqs))
                    s_pred = np.bincount(es.seq_idx, weights=mP_pred - mG_pred, minlength=len(seqs))
                    n_s = np.bincount(es.seq_idx, minlength=len(seqs)).astype(float)
                    d_rep_bs.append((W @ s_rep) / np.maximum(W @ n_s, 1e-12))
                    d_pred_bs.append((W @ s_pred) / np.maximum(W @ n_s, 1e-12))
                    d_rep_pt.append(dr["delta"])
                    d_pred_pt.append(dp["delta"])
                    print(f"r={r} {grp} L{Lm} q_o={q_o}: pred {dp['delta']:+.4f} replay {dr['delta']:+.4f} [{dr['lo']:+.4f},{dr['hi']:+.4f}] "
                          f"{'✓' if match else '✗'}", flush=True)
                x = np.array(Q_O)
                cr = crossing(x, np.stack(d_rep_bs, axis=1), np.array(d_rep_pt))
                cp = crossing(x, np.stack(d_pred_bs, axis=1), np.array(d_pred_pt))
                curves.append(dict(r=r, group=grp, M_median=M_med, Lmax=Lm, qo_star_replay=cr, qo_star_pred=cp,
                                   pred_inside_replay_ci=(None if cr["ci"] is None or not np.isfinite(cp["x_star"])
                                                          else bool(cr["ci"][0] <= cp["x_star"] <= cr["ci"][1]))))
    c = pd.DataFrame(cells)
    by_r = {str(r): dict(n=int(len(g)), n_match=int(g.match.sum()), n_sign_pred_plus=int((g.sign_pred == "+").sum()),
                         n_replay_plus=int((g.sign_replay == "+").sum()), n_replay_minus=int((g.sign_replay == "-").sum()),
                         max_abs_dev=float((g.delta_pred - g.delta_replay).abs().max()))
            for r, g in c.groupby("r")}
    main_grid = c[c.q_o.isin([0.005, 0.01, 0.02, 0.05])]
    out = dict(design=dict(q_o=Q_O, Lmax=LMAX, K_seeds=K_SEEDS, tol=TOL, split="train", simulated=True),
               by_r=by_r, total=dict(n=int(len(c)), n_match=int(c.match.sum())),
               by_r_old_grid={str(r): dict(n=int(len(g)), n_match=int(g.match.sum())) for r, g in main_grid.groupby("r")},
               cells=cells, qo_star=curves)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(by_r, indent=1))
    for cv in curves:
        print(cv["r"], cv["group"], cv["Lmax"], "q_o* replay", cv["qo_star_replay"]["x_star"], cv["qo_star_replay"]["ci"],
              "pred", cv["qo_star_pred"]["x_star"], cv["qo_star_pred"]["ci"], cv["pred_inside_replay_ci"])


if __name__ == "__main__":
    main()
