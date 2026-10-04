"""P3b E1–E5 [POST HOC — không thuộc họ xác nhận]. Đọc dump/cue/sự kiện; KHÔNG đổi θ*, c, ô hay nhãn của họ xác nhận.

E1 độ khả thi iso-KPI: sàn periodic miss_P(S = 1), sàn gate min_θ miss_G (lưới θ TRAIN w = 5, R = 30), ε_feasible = sàn periodic;
   iso-KPI (a_G − a_P(ε)) ở ε ∈ {ε_feasible + 2 pt, 10 %, 15 %, 20 %} tại θ* với CI bootstrap theo chuỗi (B = 1000, seed 42);
   tỉ lệ đúng họ xác nhận khi bỏ 24 ô H2.
E2 recall theo khung kể từ onset r(k), k = 0…10 (score ≥ 0,25, IoU ≥ 0,5), yolo26s_1024 / yolo26n_640, TRAIN / TEST, border / interior;
   kích thước box GT trung vị √(w·h) theo k.
E3 q_o* quan sát (cue mật độ tổng hợp kiểu P2B-A1: q_in = 1 trong cửa sổ khởi phát w = L, ngoài: 1 − (1 − q_o)^{M_t}, R = ∞, 3 seed;
   detector THẬT yolo26s_1024) theo nhóm mật độ TEST (tertile M trung vị chuỗi) vs điểm Thm P3-iii (ρ1 TRAIN đóng băng, q_b = 0, c = 0, r = 1).
E4 hiệu chuẩn Δ_pred vs Δ̂ (48 ô H3/H5 + H8 của P3, đọc results/p3/cells_test.csv — không tính lại).
E5 trôi kênh TRAIN → TEST tại θ* đóng băng: q_in, q_out (w = 3, 5, 10), J, a_G dự đoán; λ, ρ_onset(w); theo ego và theo tầng.
Xuất results/p3/exploratory/{e1_feasibility,e2_recall_k,e3_qostar,e4_calibration,e5_drift}.json.
Chạy: python code/p3/exploratory/p3b_explore.py [--only e1,e2,...]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "p3"))
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
from paths import CUES_DIR  # noqa: E402
from theory_checks import a_gate  # noqa: E402
from p2_engine import (EventSet, Timeline, activation, gate_or_refresh, load_events, onset_mask, periodic,  # noqa: E402
                       periodic_iso)
from p2_stats import boot_weights  # noqa: E402
from p3_replay_test import LOW, MAIN, boot_iso_diff, ego_map, hits_level, iso_grid, miss_from_lat, seq_runs  # noqa: E402
import p3b_lib as lib  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p3" / "exploratory"
STRATA = [("hovering", "6-20"), ("hovering", ">20"), ("moving", "6-20")]
EPS_FIX = [0.10, 0.15, 0.20]
QO_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
SEEDS = [42, 43, 44]


class Ctx:
    """Bối cảnh một split: khung, sự kiện (E3 cửa sổ sinh + mọi onset E3), ego, GT xe, hits theo mức."""

    def __init__(self, split, levels=(MAIN,)):
        fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
        self.fr = fr[fr.split == split]
        self.n_frames = self.fr.groupby("seq").frame.max().to_dict()
        self.ego, _ = ego_map(split)
        self.E3 = load_events(split)["E3"].copy()
        self.E3["ego"] = self.E3.seq.map(self.ego)
        e3 = pd.read_parquet(P0 / "events_E3_uavdt.parquet")
        self.e3_all = e3[(~e3.never_visible) & (e3.vis_def == "main") & (e3.split == split)]
        rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group", "visible"])
        rows = rows[(rows.group == "VEHICLE") & (rows.split == split)]
        self.rows = rows[rows.frame <= rows.seq.map(self.n_frames)]
        self.hits = {lv: hits_level(lv, self.rows, sorted(self.n_frames)) for lv in levels}
        self.split = split

    def tl(self, seqs):
        return Timeline({s: self.n_frames[s] for s in seqs})

    def seqs(self, ego=None):
        return sorted(s for s in self.n_frames if ego is None or self.ego.get(s) == ego)

    def scores(self, cue, seqs):
        """{seq: điểm cue theo khung (mảng L)}."""
        out = {}
        for s in seqs:
            t = pd.read_parquet(CUES_DIR / f"uavdt_{s}.parquet")
            t = t[(t.cue == cue) & (t.frame <= self.n_frames[s])]
            a = np.zeros(self.n_frames[s])
            a[t.frame.values - 1] = t.score.values
            out[s] = a
        return out


def frozen():
    cells = json.loads((ROOT / "results" / "p2" / "prereg_cells.json").read_text(encoding="utf-8"))["cells"]
    ch = json.loads((ROOT / "results" / "p2" / "channel_train.json").read_text(encoding="utf-8"))
    conf = [c for c in cells if c["confirmatory"]]
    cues = {}
    for c in conf:
        cues.setdefault(c["cue"], dict(theta=c["theta"], c=c["c"], w=c["w"]))
    grid = {cue: sorted({r["theta"] for r in ch["grid"] if r["cue"] == cue and r["w"] == 5}) for cue in cues}
    return cells, conf, ch, cues, grid


# ------------------------------------------------------------------ E1
def e1(T):
    cells, conf, ch, cues, grid = frozen()
    sc = json.loads((ROOT / "results" / "p3" / "score_test.json").read_text(encoding="utf-8"))
    out = dict(title=f"{lib.POST_HOC} E1 độ khả thi iso-KPI (H2 rỗng)", strata=[], iso=[])
    for eg in ("hovering", "moving"):
        seqs = T.seqs(eg)
        tl = T.tl(seqs)
        W = boot_weights(len(seqs))
        evs = T.E3[T.E3.seq.isin(seqs)].reset_index(drop=True)
        frames_seq = tl.L.astype(float)
        sc_cue = {cue: np.concatenate([v for _, v in sorted(T.scores(cue, seqs).items())]) for cue in cues}
        for mb in ("6-20", ">20"):
            if (eg, mb) not in STRATA:
                continue
            sel = (evs.M_bin == mb).values
            es = EventSet(evs[sel], T.hits[MAIN], tl)
            run1 = periodic(tl, 1.0)
            for L in (3, 5):
                floor_P = float(miss_from_lat(es.first_latency(run1), L).mean())
                gate_min = {}
                for cue, info in cues.items():
                    best = None
                    for th in grid[cue] + [info["theta"]]:
                        runG = gate_or_refresh(sc_cue[cue] >= th, tl, 30)
                        mG = float(miss_from_lat(es.first_latency(runG), L).mean())
                        aG = float(activation(runG).mean()) + info["c"]
                        if best is None or mG < best["miss_G"]:
                            best = dict(theta=th, miss_G=mG, a_G=aG)
                    gate_min[cue] = best
                pred = {c["cue"]: c["a_P_iso"] for c in conf if c["comparison"] == "iso_kpi" and c["ego"] == eg and c["M_bin"] == mb
                        and c["L"] == L}
                out["strata"].append(dict(ego=eg, M_bin=mb, L=L, n_events=int(sel.sum()), miss_P_floor_S1=floor_P, eps_feasible=floor_P,
                                          gate_floor_by_cue=gate_min, gate_floor=min(v["miss_G"] for v in gate_min.values()),
                                          pred_a_P_iso_2pct=pred, obs_a_P_iso_2pct=math.inf))
                act, ms, nev = iso_grid(tl, es, L) + (np.bincount(es.seq_idx, minlength=len(seqs)).astype(float),)
                for eps in [round(floor_P + 0.02, 4)] + EPS_FIX:
                    S_iso, aP, _ = periodic_iso(tl, es, eps, "lat", L)
                    aP = math.inf if aP is None else aP
                    for cue, info in sorted(cues.items()):
                        runG = gate_or_refresh(sc_cue[cue] >= info["theta"], tl, 30)
                        aG = float(activation(runG).mean()) + info["c"]
                        mGv = miss_from_lat(es.first_latency(runG), L)
                        mG = float(mGv.mean())
                        bd = boot_iso_diff(W, frames_seq, seq_runs(tl, runG), info["c"], act, ms, nev, eps)
                        fin = np.where(np.isfinite(bd), bd, -1e9)
                        mG_b = (W @ np.bincount(es.seq_idx, weights=mGv, minlength=len(seqs))) / np.maximum(W @ nev, 1e-12)
                        lo, hi = float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))
                        out["iso"].append(dict(ego=eg, M_bin=mb, L=L, eps=eps, eps_kind=("feasible+2pt" if eps not in EPS_FIX else "fixed"), cue=cue,
                                               a_G=aG, miss_G=mG, gate_meets=bool(mG <= eps), share_gate_meets_boot=float((mG_b <= eps).mean()),
                                               a_P_iso=aP, S_P_iso=S_iso, diff=aG - aP, diff_lo=(None if lo <= -1e8 else lo), diff_hi=(None if hi <= -1e8 else hi),
                                               share_aP_inf_boot=float(np.isinf(bd).mean()),
                                               sign=("−" if hi < 0 else "+" if lo > 0 else "0 (CI chứa 0)"),
                                               gate_cheaper=bool(mG <= eps and aG < aP)))
                print(f"E1 {eg} {mb} L{L}: floor_P={floor_P:.3f} gate_floor={out['strata'][-1]['gate_floor']:.3f}", flush=True)
    n_ok, n_dec = round(sc["three_way"]["correct"] * 72), sc["n_decided"]
    out["share_correct_all"] = dict(correct=n_ok, decided=n_dec, share=n_ok / n_dec)
    out["share_correct_excl_H2"] = dict(correct=n_ok - 24, decided=n_dec - 24, share=(n_ok - 24) / (n_dec - 24))
    return out


# ------------------------------------------------------------------ E2
def e2():
    ch = json.loads((ROOT / "results" / "p2" / "channel_train.json").read_text(encoding="utf-8"))
    out = dict(title=f"{lib.POST_HOC} E2 recall theo khung kể từ onset r(k)", r_model=ch["r_levels"], rows=[], size=[])
    for split in ("train", "test"):
        C = Ctx(split, (MAIN, LOW))
        vis = C.rows[C.rows.visible]
        vis_set = set(zip(vis.seq, vis.tid, vis.frame))
        ev = C.E3
        for lv in (MAIN, LOW):
            hs = set()
            for s, h in C.hits[lv].items():
                hs.update(zip([s] * len(h), h.tid.values, h.frame.values))
            df = pd.DataFrame(lib.recall_by_k(ev, hs, vis_set), columns=["seq", "e3_type", "k", "n", "hit"])
            for typ in ("all", "border", "interior"):
                d = df if typ == "all" else df[df.e3_type == typ]
                for k, (r, lo, hi, n) in lib.pooled_boot(d, "k").items():
                    out["rows"].append(dict(split=split, level=lv, e3_type=typ, k=int(k), r=r, lo=lo, hi=hi, n=n))
        sz = vis.assign(sz=np.sqrt(vis.w * vis.h)).set_index(["seq", "tid", "frame"]).sz
        for typ in ("all", "border", "interior"):
            e = ev if typ == "all" else ev[ev.e3_type == typ]
            for k in range(11):
                keys = [(s, int(t), int(st) + k) for s, t, st, D in zip(e.seq, e.track_id, e.start, e.D) if k < D]
                v = sz.reindex(keys).dropna()
                out["size"].append(dict(split=split, e3_type=typ, k=k, sqrt_area_median=float(v.median()) if len(v) else None, n=int(len(v))))
        print("E2", split, "xong", flush=True)
    return out


# ------------------------------------------------------------------ E3
def e3(T, e1res):
    ch = json.loads((ROOT / "results" / "p2" / "channel_train.json").read_text(encoding="utf-8"))
    rho1 = ch["rho_onset"]["1"]
    seqs = T.seqs()
    seq_M = T.fr.groupby("seq").M.median().reindex(seqs)
    terc = pd.qcut(seq_M.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    out = dict(title=f"{lib.POST_HOC} E3 q_o* quan sát vs Thm P3-iii", note="no frozen predicted CI → not a criterion",
               rho1_train=rho1, qo_grid=QO_GRID, seeds=SEEDS, groups=[])
    floor_all = {}
    for grp in ("low", "mid", "high"):
        gs = [s for s in seqs if terc[s] == grp]
        tl = T.tl(gs)
        W = boot_weights(len(gs))
        Mt = np.concatenate([T.fr[T.fr.seq == s].sort_values("frame").M.values for s in gs])
        M_med = float(np.median(Mt))
        ev = T.E3[T.E3.seq.isin(gs)].reset_index(drop=True)
        es = EventSet(ev, T.hits[MAIN], tl)
        ons = T.e3_all[T.e3_all.seq.isin(gs)][["seq", "start"]]
        nev = np.bincount(es.seq_idx, minlength=len(gs)).astype(float)
        fw = W @ tl.L.astype(float)
        for L in (3, 5):
            floor = float(miss_from_lat(es.first_latency(periodic(tl, 1.0)), L).mean())
            floor_all[(grp, L)] = floor
            im = onset_mask(tl, ons, L)
            act, ms = iso_grid(tl, es, L)
            aG_q, mG_q, runs_q, msum_q = [], [], [], []
            for q_o in QO_GRID:
                qt = 1 - (1 - q_o) ** Mt
                runs, msum, a_, m_ = np.zeros(len(gs)), np.zeros(len(gs)), [], []
                for sd in SEEDS:
                    fire = im | (np.random.default_rng(sd).random(tl.G) < qt)
                    runG = gate_or_refresh(fire, tl, math.inf)
                    mv = miss_from_lat(es.first_latency(runG), L)
                    a_.append(float(activation(runG).mean()))
                    m_.append(float(mv.mean()))
                    runs += seq_runs(tl, runG) / len(SEEDS)
                    msum += np.bincount(es.seq_idx, weights=mv, minlength=len(gs)) / len(SEEDS)
                aG_q.append(np.mean(a_)); mG_q.append(np.mean(m_)); runs_q.append(runs); msum_q.append(msum)
            for eps in (0.02, round(floor + 0.02, 4)):
                S, aP, _ = periodic_iso(tl, es, eps, "lat", L)
                aP = math.inf if aP is None else aP
                adv = [(aP - a) if m <= eps else -1.0 for a, m in zip(aG_q, mG_q)]
                adv = [1.0 if (math.isinf(x) and x > 0) else x for x in adv]  # periodic không khả thi, gate khả thi → gate thắng
                qs_obs = lib.first_cross(QO_GRID, adv)
                # bootstrap theo chuỗi
                missP = (ms @ W.T) / np.maximum(nev @ W.T, 1e-12)
                ok = missP <= eps + 1e-12
                fb = np.where(ok.all(axis=0), len(act), np.argmin(ok, axis=0))
                aP_b = np.where(fb > 0, ((act @ W.T) / fw[None, :])[np.maximum(fb - 1, 0), np.arange(len(W))], np.inf)
                qs_b = []
                for b in range(len(W)):
                    ab = [(W[b] @ r) / fw[b] for r in runs_q]
                    mb_ = [(W[b] @ m) / max(W[b] @ nev, 1e-12) for m in msum_q]
                    advb = [(1.0 if math.isinf(aP_b[b]) else aP_b[b] - a) if m <= eps else -1.0 for a, m in zip(ab, mb_)]
                    qs_b.append(lib.first_cross(QO_GRID, advb))
                qs_b = np.array(qs_b)
                fin = qs_b[np.isfinite(qs_b)]
                out["groups"].append(dict(group=grp, n_seq=len(gs), M_median=M_med, n_events=int(es.n), L=L, eps=eps,
                                          eps_kind="frozen 2 %" if eps == 0.02 else "floor+2pt", miss_P_floor=floor, a_P_iso=aP,
                                          a_G_by_qo=aG_q, miss_G_by_qo=mG_q, adv_by_qo=adv, qo_star_obs=qs_obs,
                                          qo_star_obs_ci=([float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.5 * len(qs_b) else None),
                                          share_boot_inf=float(np.isinf(qs_b).mean()), share_boot_zero=float((qs_b == 0).mean()),
                                          qo_star_pred=lib.qo_star_pred(eps, L + 1, rho1, M_med)))
                print(f"E3 {grp} L{L} eps{eps}: obs {qs_obs} pred {out['groups'][-1]['qo_star_pred']:.4g}", flush=True)
    return out


# ------------------------------------------------------------------ E4
def e4():
    d = pd.read_csv(ROOT / "results" / "p3" / "cells_test.csv")
    d = d[d.group.isin(["H3-H5", "H8"])].copy()
    out = dict(title=f"{lib.POST_HOC} E4 hiệu chuẩn Δ_pred vs Δ̂", source="results/p3/cells_test.csv (P3, không tính lại)")
    out["all48"] = lib.calibration(d.pred_value, d.obs_value)
    out["H3H5"] = lib.calibration(d[d.group == "H3-H5"].pred_value, d[d.group == "H3-H5"].obs_value)
    out["H8"] = lib.calibration(d[d.group == "H8"].pred_value, d[d.group == "H8"].obs_value)
    d["obs_sign_point"] = np.where(d.obs_value > 0.005, "+", "<=0")
    out["sign_table"] = {f"{g}|pred {p}|obs {o}": int(n) for (g, p, o), n in d.groupby(["group", "sign_pred", "obs_sign_point"]).size().items()}
    out["sign_agree_point"] = float((d.obs_sign_point == d.sign_pred).mean())
    h = d[d.group == "H3-H5"]
    out["H3H5_underestimate_loss"] = dict(n_obs_below_pred=int((h.obs_value < h.pred_value).sum()), n=int(len(h)),
                                          median_obs_minus_pred=float((h.obs_value - h.pred_value).median()))
    ex = h[(h.cue == "border_band") & (h.ego == "hovering") & (h.M_bin == "6-20") & (h.kpi.astype(str) == "5")].iloc[0]
    out["example_L5_border_hov_6_20"] = dict(pred=float(ex.pred_value), obs=float(ex.obs_value), lo=float(ex.lo), hi=float(ex.hi))
    out["rows"] = d[["id", "group", "kpi", "ego", "M_bin", "cue", "sign_pred", "pred_value", "obs_value", "lo", "hi", "verdict"]].to_dict("records")
    return out


# ------------------------------------------------------------------ E5
def e5(T):
    cells, conf, ch, cues, grid = frozen()
    out = dict(title=f"{lib.POST_HOC} E5 trôi kênh TRAIN → TEST tại θ* đóng băng", splits={}, cue_rows=[], strata_rows=[])
    for split in ("train", "test"):
        C = T if split == "test" else Ctx("train", ())
        seqs = C.seqs()
        G = sum(C.n_frames[s] for s in seqs)
        onsets = C.e3_all.groupby("seq").start.apply(list).to_dict()
        masks = {w: lib.window_mask({s: C.n_frames[s] for s in seqs}, onsets, w) for w in (1, 3, 5, 10)}
        rho = {w: float(sum(m[s].sum() for s in seqs) / G) for w, m in masks.items()}
        out["splits"][split] = dict(n_seq=len(seqs), n_frames=G, n_onsets=int(len(C.e3_all)), lambda_per_frame=len(C.e3_all) / G,
                                    rho_onset={str(k): v for k, v in rho.items()}, n_E3_birth_window=int(len(C.E3)))
        for cue, info in sorted(cues.items()):
            sc = C.scores(cue, seqs)
            fire = {s: v >= info["theta"] for s, v in sc.items()}
            for ego in ("all", "hovering", "moving"):
                ss = [s for s in seqs if ego == "all" or C.ego.get(s) == ego]
                for w in (3, 5, 10):
                    qi, qo, nin, nout = lib.q_in_out(fire, masks[w], ss)
                    rho_e = sum(masks[w][s].sum() for s in ss) / sum(C.n_frames[s] for s in ss)
                    out["cue_rows"].append(dict(split=split, cue=cue, ego=ego, w=w, theta=info["theta"], q_in=qi, q_out=qo, J=qi - qo,
                                                rho=float(rho_e), a_G_pred=a_gate(rho_e, qi, qo, 30) + info["c"],
                                                fire_rate=float(sum(fire[s].sum() for s in ss) / sum(C.n_frames[s] for s in ss))))
            # q_in theo tầng: cửa sổ w = 5 chỉ của sự kiện E3 cửa sổ sinh thuộc tầng; q_out ngoài mọi cửa sổ, chuỗi của ego
            for eg, mb in STRATA:
                ev = C.E3[(C.E3.ego == eg) & (C.E3.M_bin == mb)]
                ons = ev.groupby("seq").start.apply(list).to_dict()
                ss = sorted(ev.seq.unique())
                mk = lib.window_mask({s: C.n_frames[s] for s in ss}, ons, 5)
                kin = sum(fire[s][mk[s]].sum() for s in ss)
                nin = sum(mk[s].sum() for s in ss)
                sse = [s for s in seqs if C.ego.get(s) == eg]
                kout = sum(fire[s][~masks[5][s]].sum() for s in sse)
                nout = sum((~masks[5][s]).sum() for s in sse)
                # bắn trong 4 khung đầu (k = 0…3) của mỗi sự kiện: P(≥ 1 lần bắn)
                p_any = np.mean([fire[s][int(st) - 1: int(st) + 3].any() for s, st in zip(ev.seq, ev.start)]) if len(ev) else None
                out["strata_rows"].append(dict(split=split, cue=cue, ego=eg, M_bin=mb, n_events=int(len(ev)), q_in_stratum=float(kin / max(nin, 1)),
                                               q_out_ego=float(kout / max(nout, 1)), p_fire_any_k0_3=None if p_any is None else float(p_any)))
        print("E5", split, "xong", flush=True)
    tr = {(r["cue"], r["w"]): r for r in ch["operating_points"] if r.get("theta_star")}
    out["train_channel_frozen_w5"] = {c: dict(q_in=v["q_in"], q_out=v["q_out"]) for (c, w), v in tr.items()}
    rep = pd.read_parquet(ROOT / "results" / "p3" / "replay_test_all.parquet")
    cl = {c["id"]: c for c in cells}
    focus = []
    for x in rep[rep.confirmatory & (rep.cue == "border_band") & (rep.comparison == "matched_cost")].to_dict("records"):
        c = cl[x["id"]]
        focus.append(dict(id=x["id"], ego=x["ego"], M_bin=x["M_bin"], kpi=x["kpi"], delta_obs=x["delta"], lo=x["lo"], hi=x["hi"],
                          miss_P_obs=x["miss_P"], miss_G_obs=x["miss_G"], miss_P_pred=c["miss_P_pred"], miss_G_pred=c["miss_G_pred"],
                          delta_pred=c["delta_pred"], a_G_obs=x["a_G"], a_G_pred=c["a_cost"]))
    out["border_band_cells"] = focus
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e1,e2,e3,e4,e5")
    a = ap.parse_args()
    todo = a.only.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = Ctx("test") if {"e1", "e3", "e5"} & set(todo) else None
    res = {}
    if "e1" in todo:
        res["e1_feasibility"] = e1(T)
    if "e2" in todo:
        res["e2_recall_k"] = e2()
    if "e3" in todo:
        res["e3_qostar"] = e3(T, res.get("e1_feasibility"))
    if "e4" in todo:
        res["e4_calibration"] = e4()
    if "e5" in todo:
        res["e5_drift"] = e5(T)
    for k, v in res.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"xong {list(res)} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
