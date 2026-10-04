"""P2B-A1 (phần ranh giới): q_o* theo nghĩa ISO-KPI.

Lý do (xem mstar_exact.json): ở matched cost với q_in = 1 và r = 1, gate bắt chắc khung khởi phát ⇒ miss_G = 0 ⇒ Δ ≥ 0 theo cấu tạo;
khi q_o tăng cả hai chính sách tiến về miss ≈ 0 (hoà) — Δ không đổi dấu, nên "q_o* = điểm Δ đổi dấu" không tồn tại. Ranh giới có nghĩa là
ISO-KPI (THEORY Cor. G5, Thm P3-iii): ở latency-miss mục tiêu ε, gate khả thi (miss_G ≤ ε) và rẻ hơn (a_G < a_P(ε)) hay không.
  adv(q_o) = a_P(ε) − a_G(q_o) nếu miss_G(q_o) ≤ ε, ngược lại −1 (không khả thi);  q_o* = điểm adv đổi dấu (+ → ≤ 0).
Dự đoán (công thức): a_G = mean_t q_t; miss_G,e timeline-exact; a_P(ε) = 1/S*, S* thực lớn nhất với mean Lemma1(min(D,L+1), S, r) ≤ ε
  (bisection). CI 95%: bootstrap theo chuỗi B=1000 seed=42 (lấy mẫu lại chuỗi → F_D, khung, sự kiện).
Quan sát (replay): a_G, miss_G trung bình K=5 seed cue; a_P(ε) từ engine (periodic_iso, S thực, 20 pha). Điểm, không CI.
ε ∈ {2 %, 5 %} (chính; ε = 1 % do đuôi D = 1 quyết định — xem tail_analysis.json). Lmax ∈ {3, 5}; r ∈ {1, 0.8, 0.5}; nhóm M: tertile.
Xuất results/p2/sim/mstar_iso.json. Chạy: python code/p2/replay/p2_mstar_iso.py
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
from theory_checks import lemma_miss, q_out_eff  # noqa: E402

from p2_dump_schema import DUMPS  # noqa: E402
from p2_engine import (EventSet, Timeline, activation, gate_or_refresh, kpis, load_events, onset_frames, onset_mask,  # noqa: E402
                       periodic_iso)
from p2_match import hits_for_dump  # noqa: E402
from p2_mstar_check import K_SEEDS, pred_event_vectors, seed_of  # noqa: E402
from p2_sim_detector import dump_name  # noqa: E402
from p2_stats import B, SEED  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p2" / "sim" / "mstar_iso.json"
RS = [1.0, 0.8, 0.5]
Q_O = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
LMAX = [3, 5]
EPS = [0.02, 0.05]


def s_star_bisect(T, r, eps, S_max=400.0, iters=40):
    f = lambda S: float(np.mean(lemma_miss(T, S, r)))  # noqa: E731
    if f(1.0) > eps + 1e-12:
        return None
    if f(S_max) <= eps:
        return S_max
    lo, hi = 1.0, S_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) <= eps + 1e-12 else (lo, mid)
    return lo


def first_cross(x, adv):
    """điểm adv đổi dấu + → ≤ 0 (nội suy); inf nếu luôn > 0; 0 (hoặc nan) nếu ngay đầu ≤ 0."""
    for i in range(len(x) - 1):
        if adv[i] > 0 and adv[i + 1] <= 0:
            return x[i] + (x[i + 1] - x[i]) * adv[i] / (adv[i] - adv[i + 1])
    return math.inf if adv[0] > 0 else 0.0


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
    rng = np.random.default_rng(SEED)
    res = []
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
            ns = len(seqs)
            boot_idx = rng.integers(0, ns, size=(B, ns))
            seq_of_g = tl.seq_of_g
            for Lm in LMAX:
                im = onset_mask(tl, ons[ons.seq.isin(seqs)], Lm)
                per_q = []
                for q_o in Q_O:
                    qt = np.where(im, 1.0, q_out_eff(0.0, q_o, Mg))
                    mG, T = pred_event_vectors(es, qt, Lm, r)
                    q_seq = np.bincount(seq_of_g, weights=qt, minlength=ns)
                    # replay
                    aG_rep, mG_rep = [], np.zeros(es.n)
                    for k in range(K_SEEDS):
                        u = np.random.default_rng(seed_of(r, grp, Lm, q_o, k)).random(tl.G)
                        run = gate_or_refresh(u < qt, tl, math.inf)
                        aG_rep.append(float(activation(run).mean()))
                        mG_rep += 1 - kpis(es.first_latency(run), (Lm,))[f"L{Lm}"]
                    per_q.append(dict(q_o=q_o, qt_seq=q_seq, mG=mG, T=T, aG_rep=float(np.mean(aG_rep)), mG_rep=float((mG_rep / K_SEEDS).mean())))
                for eps in EPS:
                    # ---- dự đoán (điểm + bootstrap)
                    def adv_curve(idx):
                        cnt = np.bincount(idx, minlength=ns).astype(float)
                        ev_w = cnt[es.seq_idx]
                        L_frames = (cnt * tl.L).sum()
                        T_all = per_q[0]["T"]
                        # F_D có trọng số = lặp sự kiện theo số lần chuỗi được chọn
                        Trep = np.repeat(T_all, ev_w.astype(int))
                        S = s_star_bisect(Trep, r, eps) if len(Trep) else None
                        aP = (1.0 / S) if S else math.inf
                        out = []
                        for pq in per_q:
                            aG = float((cnt * pq["qt_seq"]).sum() / L_frames)
                            mGv = float((ev_w * pq["mG"]).sum() / max(ev_w.sum(), 1))
                            out.append((aP - aG) if mGv <= eps + 1e-12 else -1.0)
                        return np.array(out), aP
                    adv_pt, aP_pt = adv_curve(np.arange(ns))
                    cross_bs = []
                    for b in range(B):
                        a_b, _ = adv_curve(boot_idx[b])
                        cross_bs.append(first_cross(Q_O, a_b))
                    cross_bs = np.array(cross_bs)
                    fin = cross_bs[np.isfinite(cross_bs)]
                    q_pred = first_cross(Q_O, adv_pt)
                    ci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.5 * B else None
                    # ---- quan sát (replay)
                    S_rep, aP_rep, _ = periodic_iso(tl, es, eps, "lat", Lm)
                    aP_rep = aP_rep if aP_rep is not None else math.inf
                    adv_rep = np.array([(aP_rep - pq["aG_rep"]) if pq["mG_rep"] <= eps + 1e-12 else -1.0 for pq in per_q])
                    q_obs = first_cross(Q_O, adv_rep)
                    inside = None if ci is None or not np.isfinite(q_obs) else bool(ci[0] <= q_obs <= ci[1])
                    res.append(dict(r=r, group=grp, M_median=float(np.median(Mg)), Lmax=Lm, eps=eps, n_events=es.n, n_seq=ns,
                                    a_P_pred=aP_pt, a_P_replay=aP_rep, adv_pred=adv_pt.tolist(), adv_replay=adv_rep.tolist(),
                                    qo_star_pred=q_pred, qo_star_pred_ci=ci, share_boot_inf=float(np.isinf(cross_bs).mean()),
                                    qo_star_replay=q_obs, replay_inside_pred_ci=inside,
                                    a_G_replay=[pq["aG_rep"] for pq in per_q], miss_G_replay=[pq["mG_rep"] for pq in per_q]))
                    x = res[-1]
                    print(f"r={r} {grp} L{Lm} eps={eps}: aP pred {aP_pt:.3f} rep {aP_rep:.3f} | q_o* pred {q_pred:.4f} CI {ci} | replay {q_obs:.4f} inside={inside}",
                          flush=True)
    n_ok = sum(1 for x in res if x["replay_inside_pred_ci"])
    n_def = sum(1 for x in res if x["replay_inside_pred_ci"] is not None)
    OUT.write_text(json.dumps(dict(q_o=Q_O, eps=EPS, Lmax=LMAX, B=B, seed=SEED, K_seeds=K_SEEDS, n=len(res), n_defined=n_def, n_inside=n_ok,
                                   by_r={str(r): dict(n_defined=sum(1 for x in res if x["r"] == r and x["replay_inside_pred_ci"] is not None),
                                                      n_inside=sum(1 for x in res if x["r"] == r and x["replay_inside_pred_ci"]))
                                         for r in RS}, rows=res), indent=1, default=float), encoding="utf-8")
    print("inside", n_ok, "/", n_def)


if __name__ == "__main__":
    main()
