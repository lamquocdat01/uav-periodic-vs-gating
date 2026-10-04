"""P2A-A7: chạy engine replay trên UAVDT TRAIN với detector GIẢ LẬP (r × fp_rate) và cue GIẢ LẬP (A-gate).

(i)  Tái lập lý thuyết trên timeline thật (không giả định A-sparse): Lemma 1 / Cor. 1.3 (periodic), Prop. G1 (activation),
     Prop. G2 (miss gate), Thm G3 (Δ ở matched cost, r=1), Cor. G5 (onset oracle). Tiêu chí: |dự đoán − replay| < 0.01
     hoặc dự đoán nằm trong CI 95% bootstrap theo chuỗi của replay.
(ii) Bảng "detector-agnostic": r × fp × (q_in, q_out, w, R) → Δ = miss_P − miss_G ở matched cost (bisection S,
     |a_G − a_P| < 0.002) cho KPI miss (E3, E3-ROI5%) và latency (Lmax ∈ {3,5,10,30}); phân tầng M_bin; iso-KPI.
(iii) Kiểm M* (THEORY §6) bằng cue phụ thuộc mật độ q_out,t = 1 − (1 − q_o)^{M_t} (M_t = số xe visible ở khung t),
     q_in = 1, R = ∞, theo nhóm chuỗi (tertile M trung vị), matched cost trong nhóm.
KẾT QUẢ GIẢ LẬP — detector lý tưởng từ GT; không phải YOLO.
Xuất results/p2/sim/{theory_repro.json, agnostic_cells.csv, iso_kpi.csv, density_Mstar.json, oracle_stride.json, summary.json}.
Chạy: python code/p2/replay/p2_sim_run.py [--quick]
"""
import argparse
import itertools
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from theory_checks import a_gate, can_win, lemma_miss, miss_gate, q_out_eff  # noqa: E402

from p2_dump_schema import DUMPS  # noqa: E402
from p2_engine import (EventSet, Timeline, activation, gate_or_refresh, kpis, load_events, matched_periodic_S,  # noqa: E402
                       onset_frames, onset_mask, onset_oracle, oracle_per_seq_stride, periodic, periodic_iso, sim_cue)
from p2_match import hits_for_dump  # noqa: E402
from p2_sim_detector import FP_GRID, R_GRID, dump_name  # noqa: E402
from p2_stats import B, boot_mean, boot_weights, holm, paired_delta  # noqa: E402

OUT = ROOT / "results" / "p2" / "sim"
P0 = ROOT / "results" / "p0"
SPLIT = "train"
SEED = 42
Q_IN = [0.2, 0.5, 0.8, 1.0]
Q_OUT = [0.0, 0.01, 0.02, 0.05, 0.1]
W_ON = [3, 5]
R_REF = [math.inf, 30]
LMAX = [3, 5, 10, 30]
EPS = [0.05, 0.01]
Q_O = [0.005, 0.01, 0.02, 0.05]
TOL = 0.01


def cfg_seed(*parts):
    return SEED + zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def miss_vec(lat, kpi):
    k = kpis(lat, LMAX)
    return 1 - (k["det"] if kpi == "miss" else k[f"L{kpi}"])


def ci_of(values, es, W):
    pt, bs = boot_mean(values, es.seq_idx, len(es.tl.seqs), W)
    return pt, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def T_of(es, kpi):
    D = es.ev.D.values
    return D if kpi == "miss" else np.minimum(D, kpi + 1)


def agree(pred, rep, ci):
    return bool(abs(pred - rep) < TOL or (ci[0] <= pred <= ci[1]))


def refresh_phases(R, n_phase=20):
    """Cùng tập pha với p2_engine.refresh."""
    if R == math.inf:
        return None
    R = int(R)
    return np.arange(R) if R <= n_phase else np.unique(np.floor((np.arange(n_phase) + 0.5) * R / n_phase).astype(int))


def pred_gate_timeline(es, T, in_mask, q_in, q_out, R, r):
    """Miss gate kỳ vọng CHÍNH XÁC theo timeline (không A-sparse): với I_t thật, mỗi khung của cửa sổ KPI có
    xác suất thành công r nếu là khung refresh, r·q_t nếu không (q_t = q_in nếu I_t, ngược lại q_out); trung bình theo pha refresh."""
    tl = es.tl
    ph = refresh_phases(R)
    out = np.empty(es.n)
    for i, (sidx, st, Te) in enumerate(zip(es.seq_idx, es.ev.start.values, T)):
        g0 = tl.off[sidx] + st - 1
        loc = np.arange(st - 1, st - 1 + Te)
        q = np.where(in_mask[g0: g0 + Te], q_in, q_out)
        base = np.log1p(-r * q) if r * max(q_in, q_out) < 1 else None
        if ph is None:
            out[i] = np.prod(1 - r * q)
        else:
            vals = []
            for phi in ph:
                refm = (loc % int(R)) == phi
                vals.append(np.prod(np.where(refm, 1 - r, 1 - r * q)))
            out[i] = np.mean(vals)
    return out


def main(quick=False):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    fr = fr[fr.split == SPLIT]
    n_frames = fr.groupby("seq").frame.max().to_dict()
    tl = Timeline(n_frames)
    M_t = np.zeros(tl.G)
    for seq, g in fr.groupby("seq"):
        M_t[tl.g(seq, g.frame.values)] = g.M.values
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group"])
    rows = rows[(rows.group == "VEHICLE") & (rows.split == SPLIT)]
    evs = load_events(SPLIT)
    ons = onset_frames(SPLIT)
    W = boot_weights(len(tl.seqs))
    in_mask = {w: onset_mask(tl, ons, w) for w in W_ON + [1]}
    rho = {w: float(m.mean()) for w, m in in_mask.items()}
    info = dict(split=SPLIT, n_seq=len(tl.seqs), n_frames=tl.G, rho_onset=rho,
                n_events={k: int(len(v)) for k, v in evs.items()}, B=B, seed=SEED, simulated=True)
    print("info", info, flush=True)

    r_grid = [1.0] if quick else R_GRID
    fp_grid = [0.0] if quick else FP_GRID
    repro, cells, iso_rows, dens, orac = [], [], [], [], []
    for r in r_grid:
        for fp in fp_grid:
            dname = dump_name(r, fp)
            hits = hits_for_dump(DUMPS / dname, rows)
            ES = {k: EventSet(v, hits, tl) for k, v in evs.items() if k in ("E3", "E3ROI5", "E2")}
            e3 = ES["E3"]
            # ---------------- (i) Lemma 1 / Cor. 1.3 — periodic
            for S in (5, 10, 30, 60):
                lat = e3.first_latency(periodic(tl, S))
                for kpi in ["miss"] + LMAX:
                    rep, ci = ci_of(miss_vec(lat, kpi), e3, W)
                    pred = float(np.mean(lemma_miss(T_of(e3, kpi), S, r)))
                    repro.append(dict(check="Lemma1/Cor1.3 periodic", r=r, fp=fp, S=S, kpi=str(kpi), pred=pred, replay=rep,
                                      lo=ci[0], hi=ci[1], ok=agree(pred, rep, ci)))
            # ---------------- (ii) gate grid + (i) G1/G2/G3
            iso_P = {}
            for kpi in ["miss"] + LMAX[:3]:
                for eps in EPS:
                    S, aP, v = periodic_iso(tl, e3, eps, "miss" if kpi == "miss" else "lat", None if kpi == "miss" else kpi)
                    iso_P[(kpi, eps)] = (S, aP, v)
            grid = list(itertools.product(Q_IN, Q_OUT, W_ON, R_REF))
            if quick:
                grid = [g for g in grid if g[0] in (0.5, 1.0) and g[1] in (0.0, 0.02)]
            for q_in, q_out, w, R in grid:
                fire = sim_cue(tl, in_mask[w], q_in, q_out, cfg_seed(r, q_in, q_out, w, R))
                runG = gate_or_refresh(fire, tl, R)
                aG = float(activation(runG).mean())
                aG_pred = a_gate(rho[w], q_in, q_out, R)
                repro.append(dict(check="G1 activation", r=r, fp=fp, q_in=q_in, q_out=q_out, w=w, R=str(R), pred=aG_pred, replay=aG,
                                  lo=None, hi=None, ok=bool(abs(aG_pred - aG) < TOL)))
                S_m, aP = matched_periodic_S(tl, aG)
                runP = periodic(tl, S_m)
                for ename, es in ES.items():
                    latG, latP = es.first_latency(runG), es.first_latency(runP)
                    for kpi in ["miss"] + LMAX:
                        if ename == "E2" and kpi != "miss":
                            continue
                        mG, mP = miss_vec(latG, kpi), miss_vec(latP, kpi)
                        dl = paired_delta(mP, mG, es.seq_idx, len(tl.seqs), W)
                        rec = dict(r=r, fp=fp, q_in=q_in, q_out=q_out, w=w, R=str(R), event=ename, kpi=str(kpi), a_G=aG, a_P=aP, S_P=S_m,
                                   miss_G=float(mG.mean()), miss_P=float(mP.mean()), n_events=es.n, n_seq=int(len(np.unique(es.seq_idx))),
                                   M_bin="all", **dl)
                        cells.append(rec)
                        if ename == "E3" and kpi in ("miss", 3, 5):
                            T = T_of(es, kpi)
                            predS = float(np.mean([miss_gate(int(t), w, R, q_in, q_out, r) for t in T]))
                            predG = float(np.mean(pred_gate_timeline(es, T, in_mask[w], q_in, q_out, R, r)))
                            repG, ciG = ci_of(mG, es, W)
                            # (a) công thức A-sparse là CẬN TRÊN khi q_in ≥ q_out (THEORY §5.1) → kiểm một phía
                            repro.append(dict(check="G2 miss_gate A-sparse upper bound (one-sided)", r=r, fp=fp, q_in=q_in, q_out=q_out, w=w,
                                              R=str(R), kpi=str(kpi), pred=predS, replay=repG, lo=ciG[0], hi=ciG[1],
                                              ok=bool(predS >= ciG[0] - TOL), within_two_sided=agree(predS, repG, ciG)))
                            # (b) công thức chính xác theo timeline (I_t thật) → kiểm hai phía
                            repro.append(dict(check="G2 miss_gate timeline-exact", r=r, fp=fp, q_in=q_in, q_out=q_out, w=w, R=str(R), kpi=str(kpi),
                                              pred=predG, replay=repG, lo=ciG[0], hi=ciG[1], ok=agree(predG, repG, ciG)))
                            if r == 1.0:
                                predP = float(np.mean(lemma_miss(T, S_m, r)))
                                repD, ciD = ci_of(mP - mG, es, W)
                                repro.append(dict(check="G3 delta matched cost (r=1, timeline-exact gate)", r=r, fp=fp, q_in=q_in, q_out=q_out, w=w,
                                                  R=str(R), kpi=str(kpi), pred=predP - predG, replay=repD, lo=ciD[0], hi=ciD[1],
                                                  ok=agree(predP - predG, repD, ciD), pred_A_sparse=predP - predS))
                        if ename == "E3":
                            for mb in ("1-5", "6-20", ">20"):
                                sel = (es.ev.M_bin == mb).values
                                if sel.sum() == 0:
                                    continue
                                dlb = paired_delta(mP[sel], mG[sel], es.seq_idx[sel], len(tl.seqs), W)
                                cells.append(dict(rec, M_bin=mb, n_events=int(sel.sum()), n_seq=int(len(np.unique(es.seq_idx[sel]))),
                                                  miss_G=float(mG[sel].mean()), miss_P=float(mP[sel].mean()), **dlb))
                # iso-KPI: gate có đạt KPI không, và rẻ hơn periodic không
                latG = e3.first_latency(runG)
                for (kpi, eps), (S_iso, aP_iso, _) in iso_P.items():
                    mG = float(miss_vec(latG, kpi).mean())
                    met = mG <= eps + 1e-12
                    iso_rows.append(dict(r=r, fp=fp, q_in=q_in, q_out=q_out, w=w, R=str(R), kpi=str(kpi), eps=eps, miss_G=mG, gate_meets=met,
                                         a_G=aG, S_P_iso=S_iso, a_P_iso=aP_iso, gate_cheaper=bool(met and aP_iso is not None and aG < aP_iso)))
                print(f"r={r} fp={fp} q_in={q_in} q_out={q_out} w={w} R={R} aG={aG:.3f} S_P={S_m:.2f} t={time.time() - t0:.0f}s", flush=True)
            # ---------------- (i) G5 onset oracle
            for q_out in Q_OUT:
                run = onset_oracle(tl, ons, q_out, cfg_seed("oracle", r, q_out))
                a = float(activation(run).mean())
                pred_a = rho[1] + (1 - rho[1]) * q_out
                repro.append(dict(check="G5 onset oracle activation", r=r, fp=fp, q_out=q_out, pred=pred_a, replay=a, lo=None, hi=None,
                                  ok=bool(abs(pred_a - a) < TOL)))
                lat = e3.first_latency(run)
                for kpi in (3, 5):
                    rep, ci = ci_of(miss_vec(lat, kpi), e3, W)
                    # công thức chính xác theo timeline với I_t = khung onset (w=1) thật; r=1 → 0
                    pred = float(np.mean(pred_gate_timeline(e3, T_of(e3, kpi), in_mask[1], 1.0, q_out, math.inf, r)))
                    repro.append(dict(check="G5 onset oracle latency-miss", r=r, fp=fp, q_out=q_out, kpi=str(kpi), pred=pred, replay=rep,
                                      lo=ci[0], hi=ci[1], ok=agree(pred, rep, ci)))
            # ---------------- trần oracle_per_seq_stride (E3 miss) ở vài ngân sách
            if not quick:
                for a in (0.05, 0.10, 0.20):
                    o = oracle_per_seq_stride(tl, e3, a)
                    S_m, aP = matched_periodic_S(tl, a)
                    mP = float(miss_vec(e3.first_latency(periodic(tl, S_m)), "miss").mean())
                    orac.append(dict(r=r, fp=fp, a=a, oracle_a=o["a"], oracle_miss=o["miss"], periodic_miss=mP, periodic_S=S_m))
            # ---------------- (iii) cue theo mật độ → kiểm M*
            if fp == 0.0:
                seq_M = fr.groupby("seq").M.median().reindex(tl.seqs)
                terc = pd.qcut(seq_M.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
                for grp in ("low", "mid", "high"):
                    seqs_g = [s for s in tl.seqs if terc[s] == grp]
                    tl_g = Timeline({s: n_frames[s] for s in seqs_g})
                    Mg = np.concatenate([M_t[tl.off[tl.idx[s]]: tl.off[tl.idx[s]] + tl.L[tl.idx[s]]] for s in seqs_g])
                    ons_g = ons[ons.seq.isin(seqs_g)]
                    e3g = EventSet(evs["E3"][evs["E3"].seq.isin(seqs_g)], hits, tl_g)
                    Wg = boot_weights(len(seqs_g))
                    for Lm, w in ((3, 3), (5, 5)):
                        im = onset_mask(tl_g, ons_g, w)
                        rho_g = float(im.mean())
                        for q_o in Q_O:
                            qo_t = q_out_eff(0.0, q_o, Mg)
                            fire = sim_cue(tl_g, im, 1.0, None, cfg_seed("dens", r, grp, Lm, q_o), q_out_frame=qo_t)
                            runG = gate_or_refresh(fire, tl_g, math.inf)
                            aG = float(activation(runG).mean())
                            S_m, aP = matched_periodic_S(tl_g, aG)
                            mG = miss_vec(e3g.first_latency(runG), Lm)
                            mP = miss_vec(e3g.first_latency(periodic(tl_g, S_m)), Lm)
                            dl = paired_delta(mP, mG, e3g.seq_idx, len(seqs_g), Wg)
                            M_med = float(np.median(Mg))
                            pred_win = can_win(Lm + 1, w, math.inf, rho_g, float(q_out_eff(0.0, q_o, M_med)), r)[0]
                            dens.append(dict(r=r, group=grp, n_seq=len(seqs_g), M_median=M_med, Lmax=Lm, w=w, q_o=q_o, rho=rho_g,
                                             a_G=aG, a_P=aP, n_events=e3g.n, miss_G=float(mG.mean()), miss_P=float(mP.mean()),
                                             theory_can_win_at_M_median=bool(pred_win), **dl))
    # ---------------- ghi
    cdf = pd.DataFrame(cells)
    conf = (cdf.n_events >= 30) & (cdf.n_seq >= 3)
    cdf["confirmatory"] = conf
    cdf["p_holm"], cdf["reject_holm"] = np.nan, False
    if conf.any():
        adj, rej = holm(cdf.loc[conf, "p_boot"].values)
        cdf.loc[conf, "p_holm"], cdf.loc[conf, "reject_holm"] = adj, rej
    cdf.to_csv(OUT / "agnostic_cells.csv", index=False)
    pd.DataFrame(iso_rows).to_csv(OUT / "iso_kpi.csv", index=False)
    rdf = pd.DataFrame(repro)
    (OUT / "theory_repro.json").write_text(json.dumps(dict(info=info, tol=TOL, n=len(rdf), n_ok=int(rdf.ok.sum()),
                                                           by_check=rdf.groupby("check").ok.agg(["size", "sum"]).reset_index().to_dict("records"),
                                                           max_abs_dev=rdf.assign(d=(rdf.pred - rdf.replay).abs()).groupby("check").d.max().to_dict(),
                                                           rows=repro), indent=1, default=float), encoding="utf-8")
    (OUT / "density_Mstar.json").write_text(json.dumps(dens, indent=1, default=float), encoding="utf-8")
    (OUT / "oracle_stride.json").write_text(json.dumps(orac, indent=1, default=float), encoding="utf-8")
    summ = summarize(cdf, pd.DataFrame(iso_rows), rdf, pd.DataFrame(dens), info)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    print("done", time.time() - t0, "s")


def summarize(cdf, idf, rdf, ddf, info):
    s = dict(info=info, simulated=True,
             theory_repro=dict(n=len(rdf), n_ok=int(rdf.ok.sum()), all_ok=bool(rdf.ok.all()),
                               max_abs_dev=float((rdf.pred - rdf.replay).abs().max())))
    c = cdf[(cdf.M_bin == "all") & (cdf.event == "E3")]
    wins = {}
    for (r, kpi), g in c.groupby(["r", "kpi"]):
        wins[f"r{r:g}|{kpi}"] = dict(n_cfg=int(len(g)), n_gate_win=int((g.sign == "+").sum()), n_periodic_win=int((g.sign == "-").sum()),
                                     n_holm_win=int(((g.sign == "+") & g.reject_holm).sum()))
    s["matched_cost_E3"] = wins
    if len(idf):
        s["iso_kpi"] = {f"r{r:g}|{k}|eps{e:g}": dict(n_cfg=int(len(g)), n_meet=int(g.gate_meets.sum()), n_cheaper=int(g.gate_cheaper.sum()),
                                                      a_P_iso=(None if g.a_P_iso.isna().all() else float(g.a_P_iso.iloc[0])))
                        for (r, k, e), g in idf.groupby(["r", "kpi", "eps"])}
    if len(ddf):
        ddf = ddf.assign(observed_win=ddf.sign == "+", observed_lose=ddf.sign == "-")
        ddf["agree"] = np.where(ddf.theory_can_win_at_M_median, ddf.sign != "-", ddf.sign != "+")
        s["density_Mstar"] = dict(n=len(ddf), n_agree=int(ddf.agree.sum()),
                                  rows=ddf[["r", "group", "M_median", "Lmax", "q_o", "delta", "lo", "hi", "sign", "theory_can_win_at_M_median", "agree"]]
                                  .to_dict("records"))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
