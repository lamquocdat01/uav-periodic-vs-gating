"""P2A-B3: ước lượng kênh cue CHỈ TRÊN TRAIN (L8) → results/p2/channel_train.json (định dạng p2_prereg_build).

Với mỗi cue × θ (lưới phân vị điểm số TRAIN) × w ∈ {3, 5, 10}:
  q_in  = P(score ≥ θ | khung trong cửa sổ khởi phát [onset, onset+w) của ≥1 xe E3)
  q_out = P(score ≥ θ | khung ngoài mọi cửa sổ)
  (q_b, q_o): MLE nhị thức cho mô hình Lemma P3.1  P(bắn | ngoài, M) = 1 − (1−q_b)(1−q_o)^M  theo khung (M = số xe visible),
             ⇔ ln(1 − q_out(M)) = ln(1−q_b) + M ln(1−q_o); tách theo ego ∈ {all, hovering, moving} khi có nhãn.
  CI 95%: bootstrap theo chuỗi B=1000 seed=42 (q_in, q_out mọi điểm lưới; q_b, q_o cho operating point đã chọn).
Operating point mỗi (cue, w): θ cực đại q_in − q_out (Youden), cộng θ nhỏ nhất có q_out ≤ 0.01 và ≤ 0.05 — CHỌN TRÊN TRAIN.
Ở w = 5 luôn thêm θ* theo luật B1 (p2_prereg_build.theta_star: Youden với a_G(R=30) ≤ 0,25) và đánh dấu theta_star=True.
c = median(t_ms cue) / t_ms detector chính (--det-ms, hoặc đọc từ bench_openvino.json, --det-device mặc định GPU = iGPU).
  P2E: nếu có --c-json (p2_bench_c_real.py: khung TRAIN thật, median ≥ 3 lần, mẫu số yolo26s@1024 GPU) thì c lấy từ đó
  (cue OpenCV: cv2 1 luồng; tiny_det: yolo26n@320 CPU) — ghi c_source.
Nhãn ego (hovering/moving): ngưỡng ĐỀ XUẤT TỪ PHÂN BỐ trong results/p0/hover_threshold.json (UAVDT); nếu "unimodal"
  (không đề xuất) → không tách ego (chỉ "all"), không dùng ngưỡng chọn tay.
MỌI hàm ước lượng nhận `split` và assert split == "train".
P2F (01-10-2026, v2): r_levels đọc ở score_min chính 0,25 (detector_recall_train.json); sự kiện E3 = tập đầy đủ
  (events_E3_uavdt.parquet; P2G-Q5: luật lọc annotation-late của P2F bị rút); ghi det_ms_sessions = mẫu số yolo26s@1024 iGPU qua các phiên đo (chỉ để báo dải; c dùng
  mẫu số CÙNG phiên với tử số = bench_c_real). Bản v1 (r ở score 0,05): channel_train_score005.json (phụ lục).
Chạy: python code/p2/p2_estimate_channel.py --dataset uavdt [--det-ms 123.4] [--det-device GPU] [--c-json PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P0 = ROOT / "results" / "p0"
sys.path.insert(0, str(ROOT / "code"))
from paths import CUES_DIR  # noqa: E402
sys.path.insert(0, str(HERE))
from p2_prereg_build import W_STAR, theta_star  # noqa: E402
OUT = ROOT / "results" / "p2" / "channel_train.json"
B, SEED = 1000, 42
W_LIST = [3, 5, 10]
Q_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]


def _check(split):
    assert split == "train", f"L8: chỉ ước lượng trên TRAIN (nhận split={split!r})"


def in_window_mask(n_frames, onsets, w, split):
    """{seq: bool[L]} khung (1-based → chỉ số 0) trong ≥1 cửa sổ [onset, onset+w)."""
    _check(split)
    out = {}
    for seq, L in n_frames.items():
        m = np.zeros(L, bool)
        for s in onsets.get(seq, []):
            m[int(s) - 1: min(int(s) - 1 + w, L)] = True
        out[seq] = m
    return out


def fit_qb_qo(M, fire, split):
    """MLE (q_b, q_o) cho P(fire | M) = 1 − (1−q_b)(1−q_o)^M. M, fire: mảng theo khung (ngoài cửa sổ)."""
    _check(split)
    vals, inv = np.unique(M, return_inverse=True)
    n = np.bincount(inv).astype(float)
    k = np.bincount(inv, weights=fire.astype(float))

    def nll(z):
        lb, lo = -np.logaddexp(0, -z[0]), -np.logaddexp(0, -z[1])  # log q
        qb, qo = np.exp(lb), np.exp(lo)
        log_s = np.log1p(-min(qb, 1 - 1e-12)) + vals * np.log1p(-min(qo, 1 - 1e-12))  # log P(không bắn)
        p = np.clip(-np.expm1(log_s), 1e-12, 1 - 1e-12)
        return -(k * np.log(p) + (n - k) * log_s).sum()

    best = None
    for z0 in ([-4.0, -4.0], [-1.0, -6.0], [-6.0, -2.0]):
        r = minimize(nll, np.array(z0), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    sig = 1 / (1 + np.exp(-best.x))
    return float(sig[0]), float(sig[1])


def estimate(traces, n_frames, onsets, M_by_seq, split, ego_by_seq=None, t_det_ms=None, theta_grid=None, B_=B, c_by_cue=None):
    """traces: {seq: DataFrame(frame, cue, score, t_ms)} — chỉ chuỗi TRAIN. Trả dict kênh."""
    _check(split)
    seqs = sorted(traces)
    rng = np.random.default_rng(SEED)
    W = np.zeros((B_, len(seqs)))
    for b in range(B_):
        W[b] = np.bincount(rng.integers(0, len(seqs), len(seqs)), minlength=len(seqs))
    ego_by_seq = ego_by_seq or {}
    total = sum(n_frames[s] for s in seqs)
    lam = sum(len(onsets.get(s, [])) for s in seqs) / total
    masks = {w: in_window_mask({s: n_frames[s] for s in seqs}, onsets, w, split) for w in W_LIST + [1]}
    rho = {w: float(sum(masks[w][s].sum() for s in seqs) / total) for w in W_LIST + [1]}
    cues = sorted(set().union(*[set(t.cue.unique()) for t in traces.values()]))
    out = dict(split=split, simulated=False, lambda_per_frame=lam, rho_onset={str(k): v for k, v in rho.items()},
               n_seq=len(seqs), n_frames=total, B=B_, seed=SEED, grid=[], operating_points=[])
    for cue in cues:
        sc = {s: traces[s][traces[s].cue == cue].sort_values("frame") for s in seqs}
        allsc = np.concatenate([d.score.values for d in sc.values()])
        thetas = theta_grid if theta_grid is not None else np.unique(np.quantile(allsc, Q_GRID))
        t_ms = float(np.median(np.concatenate([d.t_ms.values for d in sc.values()])))
        c = (c_by_cue or {}).get(cue, (t_ms / t_det_ms) if t_det_ms else None)
        for w in W_LIST:
            rows = []
            for th in thetas:
                kin, nin, kout, nout = (np.zeros(len(seqs)) for _ in range(4))
                for i, s in enumerate(seqs):
                    sco = np.zeros(n_frames[s])
                    sco[sc[s].frame.values - 1] = sc[s].score.values
                    f = sco >= th
                    m = masks[w][s]
                    kin[i], nin[i], kout[i], nout[i] = f[m].sum(), m.sum(), f[~m].sum(), (~m).sum()
                q_in, q_out = kin.sum() / max(nin.sum(), 1), kout.sum() / max(nout.sum(), 1)
                bi, bo = (W @ kin) / np.maximum(W @ nin, 1), (W @ kout) / np.maximum(W @ nout, 1)
                rows.append(dict(cue=cue, w=w, theta=float(th), q_in=float(q_in), q_out=float(q_out),
                                 q_in_ci=[float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))],
                                 q_out_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))], t_ms=t_ms, c=c))
            out["grid"] += rows
            g = pd.DataFrame(rows)
            picks = {int(g.assign(j=g.q_in - g.q_out).j.idxmax())}
            for cap in (0.01, 0.05):
                ok = g[g.q_out <= cap]
                if len(ok):
                    picks.add(int(ok.theta.idxmin()))
            ts = theta_star([dict(r, c=c or 0.0) for r in rows], rho[W_STAR]) if w == W_STAR else None
            if ts is not None:
                picks.add(next(k for k, r in enumerate(rows) if r["theta"] == ts["theta"]))
            for i in sorted(picks):
                r = rows[i]
                qb, qo, ci = {}, None, {}
                for ego in ["all"] + sorted({ego_by_seq.get(s, "unknown") for s in seqs} - {"unknown"}):
                    ss = [s for s in seqs if ego == "all" or ego_by_seq.get(s) == ego]
                    Mv, Fv, Sv = [], [], []
                    for j, s in enumerate(ss):
                        sco = np.zeros(n_frames[s])
                        sco[sc[s].frame.values - 1] = sc[s].score.values
                        m = ~masks[w][s]
                        Mv.append(M_by_seq[s][m])
                        Fv.append(sco[m] >= r["theta"])
                        Sv.append(np.full(m.sum(), j))
                    Mv, Fv, Sv = np.concatenate(Mv), np.concatenate(Fv), np.concatenate(Sv)
                    b_, o_ = fit_qb_qo(Mv, Fv, split)
                    qb[ego] = b_
                    if ego == "all":
                        qo = o_
                        # CI bootstrap theo chuỗi (lấy mẫu lại chỉ số chuỗi)
                        bs = []
                        for bb in range(min(B_, 200)):
                            pick = rng.integers(0, len(ss), len(ss))
                            sel = np.concatenate([np.flatnonzero(Sv == p) for p in pick])
                            bs.append(fit_qb_qo(Mv[sel], Fv[sel], split))
                        bs = np.array(bs)
                        ci = dict(q_b=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                                  q_o=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))], B=int(len(bs)))
                out["operating_points"].append(dict(cue=cue, theta=r["theta"], w=w, q_in=r["q_in"], q_out=r["q_out"], q_in_ci=r["q_in_ci"],
                                                    q_out_ci=r["q_out_ci"], q_b=qb, q_o=qo, qbqo_ci=ci, c=(c if c is not None else 0.0),
                                                    c_measured=c is not None, t_ms=t_ms,
                                                    theta_star=bool(ts is not None and r["theta"] == ts["theta"])))
    return out


def det_ms_from_bench(path, device="CPU", tag="yolo26s_1024"):
    """median ms (tiền xử lý + suy luận) của detector chính trong bench_openvino.json (khối "detector"); None nếu không thấy."""
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    model, _, size = tag.rpartition("_")
    for r in d.get("detector", []):
        hit = (r.get("model") == model and str(r.get("imgsz")) == size) if "model" in r else tag in str(r.get("ir", ""))
        if hit and r.get("device") == device and "total_latency_median_ms" in r:
            return float(r["total_latency_median_ms"])
    return None


C_KEYS = {"raw_diff": "raw_diff/1thr", "ego_comp_diff": "ego_comp_diff/1thr", "orb_lite_diff": "orb_lite_diff/1thr",
          "border_band": "border_band/1thr", "tiny_det": "tiny_det_yolo26n_320/CPU"}


def c_from_real(path):
    """{cue: c_median} từ bench_c_real.json (split phải là train); {} nếu không có file."""
    if not Path(path).exists():
        return {}, None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    _check(d["split"])
    return {cue: d["c"][k]["c_median"] for cue, k in C_KEYS.items() if k in d["c"]}, d["denominator"]


def ego_labels(ego_seq, hover_json, dataset="UAVDT"):
    """{seq: hovering|moving} theo ngưỡng đề xuất từ phân bố; ({}, None) nếu unimodal/không có file."""
    if ego_seq is None or not Path(hover_json).exists():
        return {}, None
    thr = json.loads(Path(hover_json).read_text(encoding="utf-8")).get("by_dataset", {}).get(dataset, {}).get("proposed_threshold_px")
    if thr is None:
        return {}, None
    e = ego_seq[ego_seq.dataset == dataset]
    return dict(zip(e.seq, np.where(e.shift_px_median < thr, "hovering", "moving"))), float(thr)


DET_SESSIONS = [  # (nhãn, nguồn) — mẫu số yolo26s@1024 GPU (pre+infer) các phiên bench trước; bản 29-09 nằm trong lịch sử git
    ("2026-09-29 bench_openvino (P2D-C1)", "git:266fa58:results/p2/bench_real/bench_openvino.json"),
    ("2026-09-30 bench_openvino (P2E-B1a)", "results/p2/bench_real/bench_openvino.json"),
]


def det_ms_sessions(c_json):
    """[{session, det_ms}] cho yolo26s@1024 GPU: các phiên bench_openvino + phiên bench_c_real (mẫu số của c)."""
    import subprocess
    out = []
    for lab, src in DET_SESSIONS:
        try:
            if src.startswith("git:"):
                _, rev, path = src.split(":", 2)
                txt = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
                tmp = ROOT / "results" / "p2" / ".det_tmp.json"
                tmp.write_text(txt, encoding="utf-8")
                v = det_ms_from_bench(tmp, "GPU")
                tmp.unlink()
            else:
                v = det_ms_from_bench(ROOT / src, "GPU")
        except Exception:  # noqa: BLE001
            v = None
        out.append(dict(session=lab, source=src, det_ms=v))
    if Path(c_json).exists():
        d = json.loads(Path(c_json).read_text(encoding="utf-8"))
        out.append(dict(session="2026-09-30 bench_c_real (P2E-B1b, mẫu số c)", source=Path(c_json).name,
                        det_ms=d["t_ms"]["yolo26s_1024/GPU"]["median"], runs=d["t_ms"]["yolo26s_1024/GPU"]["runs"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uavdt")
    ap.add_argument("--det-ms", type=float, default=None)
    ap.add_argument("--det-device", default="GPU", choices=["CPU", "GPU"])
    ap.add_argument("--c-json", default=str(ROOT / "results" / "p2" / "bench_real" / "bench_c_real.json"))
    ap.add_argument("--recall-json", default=str(ROOT / "results" / "p2" / "detector_recall_train.json"))
    ap.add_argument("--main-level", default="yolo26s_1024")
    a = ap.parse_args()
    split = "train"
    fr = pd.read_parquet(P0 / f"frames_{a.dataset}.parquet")
    fr = fr[fr.split == split]
    n_frames = fr.groupby("seq").frame.max().to_dict()
    traces = {s: pd.read_parquet(CUES_DIR / f"{a.dataset}_{s}.parquet") for s in n_frames if (CUES_DIR / f"{a.dataset}_{s}.parquet").exists()}
    if not traces:
        print("chưa có cue trace TRAIN (cần frames) → không ước lượng")
        return
    n_frames = {s: n_frames[s] for s in traces}
    # khung ngoài phạm vi GT (M0207: ảnh 572–885 không có nhãn) → bỏ khỏi ước lượng
    traces = {s: t[t.frame <= n_frames[s]] for s, t in traces.items()}
    e3 = pd.read_parquet(P0 / f"events_E3_{a.dataset}.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main") & (e3.split == split)]
    onsets = e3.groupby("seq").start.apply(list).to_dict()
    M = {s: g.sort_values("frame").M.values for s, g in fr.groupby("seq")}
    eg = P0 / "egomotion_seq.parquet"
    ego, hover_thr = ego_labels(pd.read_parquet(eg) if eg.exists() else None, P0 / "hover_threshold.json", a.dataset.upper())
    # bench weights thật (P2D/P2E) ở results/p2/bench_real/; results/p2/bench_openvino.json là bản COCO tạm của P2C
    det_ms = a.det_ms or det_ms_from_bench(ROOT / "results" / "p2" / "bench_real" / "bench_openvino.json", a.det_device)
    c_real, c_denom = c_from_real(a.c_json)
    out = estimate(traces, n_frames, onsets, M, split, ego, det_ms, c_by_cue=c_real)
    out["dataset"] = a.dataset.upper()
    rj = Path(a.recall_json)
    if rj.exists():  # P2E: recall theo khung đo trên dump TRAIN thật (p2_detector_recall.py)
        rec = json.loads(rj.read_text(encoding="utf-8"))
        _check(rec["split"])
        out["r_levels"], out["r_source"] = rec["r_levels"], f"{rj.name} (score ≥ {rec['score_min_main']}, IoU ≥ {rec['iou']})"
        out["score_min_main"] = rec["score_min_main"]
        out["r_levels_by_score"] = rec.get("r_levels_by_score", {})
        out["r_missing"] = rec.get("missing", [])
        out["main_level"] = a.main_level
    else:
        out["r_levels"] = {}
    out["det_ms"] = det_ms
    out["det_device"] = a.det_device
    out["c_source"] = {cue: (f"bench_c_real ({C_KEYS[cue]} / {c_denom})" if cue in c_real else f"cue trace t_ms / det_ms {a.det_device}")
                       for cue in sorted({op["cue"] for op in out["operating_points"]})}
    out["hover_threshold_px"] = hover_thr
    out["version"] = 2
    out["det_ms_sessions"] = det_ms_sessions(a.c_json)
    out["c_denominator"] = c_denom
    out["n_e3_train"] = int(len(e3))
    out["e3_filter"] = "không lọc — E3 đầy đủ theo định nghĩa P0 (P2G-Q5; audit khung trước onset chỉ mô tả, PREREG §5)"
    out["ego_counts"] = pd.Series(ego).value_counts().to_dict() if ego else {}
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT, len(out["operating_points"]), "operating points")


if __name__ == "__main__":
    main()
