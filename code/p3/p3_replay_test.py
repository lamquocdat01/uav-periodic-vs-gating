"""P3-B1 (+ C): replay UAVDT TEST trên dump detector THẬT + cue trace THẬT ở θ* đã đóng băng → quan sát cho p2_prereg_score.

Đầu vào đóng băng (không tính lại gì): results/p2/prereg_cells.json (sha256 kiểm với PREREG_28_FREEZE.md) — θ*, c, ô, dự đoán.
Engine: code/p2/replay (p2_engine, p2_match, p2_stats) — đóng băng, chỉ import.

Định nghĩa quan sát (theo PREREG_28_DRAFT §1–§4 và cách replay A7 p2_sim_run, áp cho dữ liệu thật):
  detector "phát hiện" = box dump score ≥ 0,25 (PREREG §2 Q1) ghép GT xe IoU ≥ 0,5 một-một (p2_match.match_seq) — như r TRAIN.
  khung: chỉ khung trong phạm vi GT (frames_uavdt.parquet); sự kiện: load_events(split)["E3"] (VISIBLE chính, cửa sổ sinh W = 120).
  tầng ego (thuộc tính CHUỖI, ngưỡng đóng băng 0,61 px) → timeline con gồm các chuỗi của tầng; M_bin (thuộc tính SỰ KIỆN) → chọn
    sự kiện trong timeline đó. Chi phí khớp trong timeline của tầng ego (dự đoán của ô dùng chi phí riêng tầng: q_b theo ego).
  gate:      run = (cue ≥ θ*) ∨ refresh(R = 30) (20 pha refresh); a_G = (#chạy)/(#khung) + c (cue tính mọi khung).
  periodic:  S thực khớp a_G (p2_engine.matched_periodic_S, |a_P − a_G| < 5e-4), trung bình 20 pha.
  miss_X,e(L) = 1 − P_pha(độ trễ onset ≤ L);  Δ̂ = mean_e(miss_P − miss_G) (> 0: gate thắng).
  H8:  ΔΔ̂ = Δ̂(r thấp) − Δ̂(r chính) trên CÙNG sự kiện, CÙNG lịch gate; CI bootstrap ghép cặp của vector theo sự kiện.
  iso-KPI (H2): a_P(ε) = activation của S lớn nhất có miss_P(L) ≤ ε (p2_engine.periodic_iso, điểm);
       gate_cheaper = (miss_G ≤ ε) ∧ (a_G < a_P(ε)); CI(a_G − a_P(ε)) bootstrap theo chuỗi trên lưới S (300 điểm hình học 1…400).
  CI 95 % = bootstrap theo chuỗi của timeline tầng, B = 1000, seed 42 (p2_stats.boot_weights / paired_delta).
Luật cỡ mẫu khi chấm (DRAFT §2 "re-checked on TEST when scoring"): ô có < 30 sự kiện hoặc < 3 chuỗi trên TEST → quan sát ghi
  CI = (−∞, +∞) ⇒ CHƯA KẾT LUẬN (ô vẫn ở trong họ, không thêm/bớt ô). Luật này ghi vào code TRƯỚC khi tính Δ̂ TEST.
Tiêu chí (ii) (q_o* ngoài CI dự đoán): prereg_cells.json không có CI dự đoán q_o* cho kênh thật → không truyền "boundary".
Xuất: results/p3/obs_test.json (quan sát họ xác nhận, định dạng p2_prereg_score), results/p3/replay_test_all.parquet (mọi ô có dự đoán,
  gồm ô mô tả), results/p3/replay_test_meta.json.
Biến thể MÔ TẢ (phần C, sau B, không ảnh hưởng B; chỉ ô họ xác nhận, iso chỉ điểm): --variant
  onset_alt  onset = onset_alt của audit TEST (start − shift, D + shift; results/p0/audit_prebirth_test.json)
  roi5       sự kiện E3ROI5 (ROI biên m = 5 %) thay E3
  score005 / score050   detector "phát hiện" ở score ≥ 0,05 / 0,50 thay 0,25
  → results/p3/replay_test_<variant>.parquet (không ghi obs_test.json).
Chạy: python code/p3/p3_replay_test.py [--split test|train] [--no-desc] [--variant main|onset_alt|roi5|score005|score050]
  --split train: chỉ kiểm khâu dựng ô (số sự kiện/chuỗi mỗi ô phải bằng n_events_train / n_seq_train đóng băng), không ghi quan sát.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
from paths import CUES_DIR, DUMPS_DIR  # noqa: E402
from p2_engine import (BIG, EventSet, Timeline, activation, gate_or_refresh, load_events, matched_periodic_S,  # noqa: E402
                       periodic, periodic_iso)
from p2_match import match_seq  # noqa: E402
from p2_stats import boot_weights, paired_delta  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p3"
CELLS = ROOT / "results" / "p2" / "prereg_cells.json"
CELLS_SHA_FREEZE = "fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e"
SCORE_MIN = 0.25
MAIN, LOW = "yolo26s_1024", "yolo26n_640"
LMAX = (3, 5, 10, 30)
S_GRID = np.unique(np.geomspace(1.0, 400.0, 300))
INF_CI = (-math.inf, math.inf)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ego_map(split):
    thr = float(json.loads((P0 / "hover_threshold.json").read_text(encoding="utf-8"))["by_dataset"]["UAVDT"]["proposed_threshold_px"])
    if split == "train":
        e = pd.read_parquet(P0 / "egomotion_seq.parquet")
        e = e[e.dataset == "UAVDT"]
    else:
        e = pd.read_parquet(OUT / "egomotion_seq_test.parquet")
    return dict(zip(e.seq, np.where(e.shift_px_median < thr, "hovering", "moving"))), thr


def hits_level(level, rows, seqs, score_min=SCORE_MIN):
    """{seq: DataFrame(tid, frame)} — detection = box dump score ≥ score_min (như detector_recall / PREREG §2)."""
    out = {}
    for s in seqs:
        d = pd.read_parquet(DUMPS_DIR / level / f"uavdt_{s}.parquet")
        g = rows[rows.seq == s]
        out[s] = match_seq(g, d[d.score >= score_min])
    return out


def cue_fire(tl, cue, theta, n_frames):
    f = np.zeros(tl.G, bool)
    for s in tl.seqs:
        t = pd.read_parquet(CUES_DIR / f"uavdt_{s}.parquet")
        t = t[(t.cue == cue) & (t.frame <= n_frames[s])]
        if t.frame.nunique() != n_frames[s]:
            raise RuntimeError(f"cue trace {cue} {s}: {t.frame.nunique()} / {n_frames[s]} khung")
        f[tl.g(s, t.frame.values)] = t.score.values >= theta
    return f


def miss_from_lat(lat, L):
    return 1.0 - (lat <= L).mean(axis=0) if L != "miss" else 1.0 - (lat < BIG).mean(axis=0)


def seq_runs(tl, run):
    """số khung chạy kỳ vọng (trung bình pha) mỗi chuỗi → (n_seq,)"""
    return np.add.reduceat(np.atleast_2d(run).mean(axis=0), tl.off)


def iso_grid(tl, es, L):
    """lưới S → (act_seq (nS × n_seq), miss_seq_sum (nS × n_seq)) cho bootstrap a_P(ε)."""
    nseq = len(tl.seqs)
    act = np.zeros((len(S_GRID), nseq))
    ms = np.zeros((len(S_GRID), nseq))
    for j, S in enumerate(S_GRID):
        run = periodic(tl, S)
        act[j] = seq_runs(tl, run)
        ms[j] = np.bincount(es.seq_idx, weights=miss_from_lat(es.first_latency(run), L), minlength=nseq)
    return act, ms


def boot_iso_diff(W, frames_seq, runsG_seq, c, act, ms, n_ev_seq, eps):
    """bootstrap (B,) của a_G − a_P(ε). a_P(ε) = activation tại S lớn nhất trên lưới trước lần vi phạm đầu (giả định đơn điệu như
    periodic_iso); S = 1 không đạt → a_P = ∞ (diff = −∞)."""
    fw = W @ frames_seq
    aG = (W @ runsG_seq) / fw + c
    miss = (ms @ W.T) / np.maximum(n_ev_seq @ W.T, 1e-12)        # nS × B
    aP_grid = (act @ W.T) / fw[None, :]                            # nS × B
    ok = miss <= eps + 1e-12
    first_bad = np.where(ok.all(axis=0), len(S_GRID), np.argmin(ok, axis=0))
    aP = np.where(first_bad > 0, aP_grid[np.maximum(first_bad - 1, 0), np.arange(W.shape[0])], np.inf)
    return aG - aP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--no-desc", action="store_true", help="chỉ ô họ xác nhận")
    ap.add_argument("--variant", default="main", choices=["main", "onset_alt", "roi5", "score005", "score050"])
    a = ap.parse_args()
    var = a.variant
    score_min = {"score005": 0.05, "score050": 0.50}.get(var, SCORE_MIN)
    if var != "main":
        a.no_desc = True
    t0 = time.time()
    sha = sha256(CELLS)
    assert sha == CELLS_SHA_FREEZE, f"prereg_cells.json sha256 {sha} ≠ FREEZE"
    allcells = json.loads(CELLS.read_text(encoding="utf-8"))["cells"]
    conf = [c for c in allcells if c["confirmatory"]]
    assert len(conf) == 72, len(conf)
    split = a.split
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    fr = fr[fr.split == split]
    n_frames = fr.groupby("seq").frame.max().to_dict()
    ego, thr = ego_map(split)
    ev = load_events(split)
    E3 = (ev["E3ROI5"] if var == "roi5" else ev["E3"]).copy()
    if var == "onset_alt":
        au = pd.DataFrame(json.loads((P0 / f"audit_prebirth_{split}.json").read_text(encoding="utf-8"))["events"])[["seq", "track_id", "shift"]]
        E3 = E3.merge(au, on=["seq", "track_id"], how="left")
        sh = E3["shift"].fillna(0).astype(int)
        E3["start"], E3["D"] = E3.start - sh, E3.D + sh
    E3["ego"] = E3.seq.map(ego)
    # kiểm khâu dựng ô trên TRAIN: số sự kiện/chuỗi mỗi tầng = số đóng băng
    if split == "train":
        bad = []
        for c in allcells:
            if c.get("pool") is not None or c["M_bin"] == "all":
                continue
            sub = E3[(E3.ego == c["ego"]) & (E3.M_bin == c["M_bin"])]
            if (len(sub), sub.seq.nunique()) != (c["n_events_train"], c["n_seq_train"]):
                bad.append((c["id"], len(sub), sub.seq.nunique(), c["n_events_train"], c["n_seq_train"]))
        print("TRAIN kiểm số sự kiện/chuỗi theo ô:", "KHỚP" if not bad else f"LỆCH {len(bad)}", bad[:5])
        return
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group"])
    rows = rows[(rows.group == "VEHICLE") & (rows.split == split)]
    rows = rows[rows.frame <= rows.seq.map(n_frames)]
    levels = [MAIN, LOW]
    hits = {lv: hits_level(lv, rows, sorted(n_frames), score_min) for lv in levels}
    print(f"hits xong {time.time() - t0:.0f}s", flush=True)
    # θ*, c theo cue (từ ô đóng băng)
    cues = {}
    for c in conf:
        cues.setdefault(c["cue"], (c["theta"], c["c"], c["w"]))
    # tầng ego: hovering, moving, all (ô pool M+ego, mô tả)
    strata = {e: sorted(s for s in n_frames if ego.get(s) == e) for e in ("hovering", "moving")}
    strata["all"] = sorted(n_frames)
    need = conf if a.no_desc else [c for c in allcells if c.get("r_level") in (MAIN, f"{MAIN}>{LOW}") or c.get("r_level") == LOW]
    res_rows, obs = [], []
    conf_ids = {c["id"] for c in conf}
    iso_cache, iso_boot_cache = {}, {}
    for eg, seqs in strata.items():
        if not seqs:
            continue
        tl = Timeline({s: n_frames[s] for s in seqs})
        W = boot_weights(len(seqs))
        evs = E3[E3.seq.isin(seqs)].reset_index(drop=True)
        ES = {lv: EventSet(evs, hits[lv], tl) for lv in levels}
        frames_seq = tl.L.astype(float)
        for cue, (theta, c_cost, w) in sorted(cues.items()):
            fire = cue_fire(tl, cue, theta, n_frames)
            for R in (30, math.inf):
                Rs = "inf" if R == math.inf else 30
                runG = gate_or_refresh(fire, tl, R)
                aG = float(activation(runG).mean()) + c_cost
                S_m, aP = matched_periodic_S(tl, aG)
                runP = periodic(tl, S_m)
                runsG_seq = seq_runs(tl, runG)
                vec = {}
                for lv in levels:
                    latG, latP = ES[lv].first_latency(runG), ES[lv].first_latency(runP)
                    for L in ("miss",) + LMAX:
                        vec[(lv, L)] = (miss_from_lat(latP, L), miss_from_lat(latG, L))
                for cl in need:
                    if cl["cue"] != cue or str(cl["R"]) != str(Rs) or cl["ego"] != eg:
                        continue
                    if cl["pool"] == "M+ego" and eg != "all":
                        continue
                    sel = np.ones(len(evs), bool) if cl["M_bin"] == "all" else (evs.M_bin == cl["M_bin"]).values
                    n_ev, n_seq = int(sel.sum()), int(evs.seq[sel].nunique())
                    size_ok = bool(n_ev >= 30 and n_seq >= 3)
                    base = dict(id=cl["id"], confirmatory=cl["id"] in conf_ids, hypothesis=cl["hypothesis"], comparison=cl["comparison"],
                                kpi=cl["kpi"], ego=eg, M_bin=cl["M_bin"], cue=cue, R=str(Rs), r_level=cl["r_level"], sign_pred=cl["sign_pred"],
                                n_events_test=n_ev, n_seq_test=n_seq, size_rule_test=size_ok, a_G=aG, S_P=S_m, a_P=aP,
                                n_events_train=cl["n_events_train"], n_seq_train=cl["n_seq_train"])
                    if n_ev == 0:
                        res_rows.append(dict(base, note="0 sự kiện TEST"))
                        if base["confirmatory"]:
                            obs.append(dict(id=cl["id"], delta=float("nan"), lo=INF_CI[0], hi=INF_CI[1], p_boot=None, size_rule_test=False))
                        continue
                    sidx = ES[MAIN].seq_idx[sel]
                    if cl["comparison"] == "matched_cost":
                        L = cl["kpi"] if cl["kpi"] == "miss" else int(cl["kpi"])
                        mP, mG = vec[(cl["r_level"], L)]
                        d = paired_delta(mP[sel], mG[sel], sidx, len(seqs), W)
                        rec = dict(base, delta=d["delta"], lo=d["lo"], hi=d["hi"], p_boot=d["p_boot"], miss_P=float(mP[sel].mean()),
                                   miss_G=float(mG[sel].mean()), pred=cl.get("delta_pred"))
                    elif cl["comparison"] == "r_pair":
                        hi_l, lo_l = cl["r_level"].split(">")
                        L = int(cl["L"])
                        if (hi_l, L) not in vec or (lo_l, L) not in vec:
                            continue
                        dlo = vec[(lo_l, L)][0] - vec[(lo_l, L)][1]
                        dhi = vec[(hi_l, L)][0] - vec[(hi_l, L)][1]
                        d = paired_delta(dlo[sel], dhi[sel], sidx, len(seqs), W)
                        rec = dict(base, delta=d["delta"], lo=d["lo"], hi=d["hi"], p_boot=d["p_boot"], delta_lo_level=float(dlo[sel].mean()),
                                   delta_hi_level=float(dhi[sel].mean()), pred=cl.get("ddelta_pred"))
                    else:  # iso_kpi
                        L, eps = int(cl["L"]), float(cl["eps"])
                        lv = cl["r_level"]
                        mG = vec[(lv, L)][1]
                        key = (eg, cl["M_bin"], lv, L, eps)
                        if key not in iso_cache:
                            es_sel = EventSet(evs[sel], hits[lv], tl)
                            S_iso, aP_iso, v = periodic_iso(tl, es_sel, eps, "lat", L)
                            iso_cache[key] = (S_iso, (aP_iso if aP_iso is not None else math.inf), v)
                        S_iso, aP_iso, _ = iso_cache[key]
                        mGs = float(mG[sel].mean())
                        met = bool(mGs <= eps + 1e-12)
                        cheaper = bool(met and aG < aP_iso)
                        rec = dict(base, eps=eps, miss_G=mGs, gate_meets=met, a_P_iso=aP_iso, S_P_iso=S_iso, gate_cheaper=cheaper,
                                   diff=aG - aP_iso, pred_a_G=cl.get("a_cost"), pred_a_P_iso=cl.get("a_P_iso"))
                        if base["confirmatory"] and var == "main":
                            gk = (eg, cl["M_bin"], lv, L)
                            if gk not in iso_boot_cache:
                                es_sel = EventSet(evs[sel], hits[lv], tl)
                                iso_boot_cache[gk] = iso_grid(tl, es_sel, L) + (np.bincount(es_sel.seq_idx, minlength=len(seqs)).astype(float),)
                            act, ms, nev = iso_boot_cache[gk]
                            bd = boot_iso_diff(W, frames_seq, runsG_seq, c_cost, act, ms, nev, eps)
                            fin = np.where(np.isfinite(bd), bd, -1e9)
                            rec.update(diff_lo=float(np.percentile(fin, 2.5)), diff_hi=float(np.percentile(fin, 97.5)),
                                       share_aP_inf_boot=float(np.isinf(bd).mean()))
                    res_rows.append(rec)
                    if base["confirmatory"] and var == "main":
                        if rec["comparison"] == "iso_kpi":
                            o = dict(id=cl["id"], gate_cheaper=rec["gate_cheaper"], diff=rec["diff"], diff_lo=rec["diff_lo"], diff_hi=rec["diff_hi"])
                            if not size_ok:
                                o.update(diff_lo=INF_CI[0], diff_hi=INF_CI[1])
                        else:
                            o = dict(id=cl["id"], delta=rec["delta"], lo=rec["lo"], hi=rec["hi"], p_boot=rec["p_boot"])
                            if not size_ok:
                                o.update(lo=INF_CI[0], hi=INF_CI[1])
                        o["size_rule_test"] = size_ok
                        obs.append(o)
                print(f"{eg} {cue} R={Rs}: a_G={aG:.4f} S_P={S_m:.2f} t={time.time() - t0:.0f}s", flush=True)
    if var != "main":
        df = pd.DataFrame(res_rows).assign(variant=var, score_min=score_min)
        df.to_parquet(OUT / f"replay_test_{var}.parquet", index=False)
        print(f"wrote replay_test_{var}.parquet {len(df)} ô, {time.time() - t0:.0f}s")
        return
    got = {o["id"] for o in obs}
    missing = sorted(conf_ids - got)
    assert not missing, f"thiếu quan sát cho {len(missing)} ô xác nhận: {missing[:3]}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "obs_test.json").write_text(json.dumps(dict(obs=obs, boundary=None,
                                                       boundary_note="không có CI dự đoán q_o* đóng băng cho kênh thật trong prereg_cells.json → tiêu chí (ii) không đánh giá"),
                                                  indent=1, default=float), encoding="utf-8")
    df = pd.DataFrame(res_rows)
    df.to_parquet(OUT / "replay_test_all.parquet", index=False)
    meta = dict(split=split, cells_sha256=sha, n_conf=len(conf), n_obs_conf=len(obs), n_rows=len(df), score_min=SCORE_MIN,
                levels=levels, hover_threshold_px=thr, ego_counts=pd.Series(ego).value_counts().to_dict(),
                n_E3_birth_window=int(len(E3)), cues={k: dict(theta=v[0], c=v[1], w=v[2]) for k, v in cues.items()},
                s_grid=dict(n=len(S_GRID), lo=1.0, hi=400.0), seconds=time.time() - t0)
    (OUT / "replay_test_meta.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
