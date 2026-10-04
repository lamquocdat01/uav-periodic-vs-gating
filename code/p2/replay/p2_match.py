"""P2A-A3: ghép detection → GT theo khung (IoU ≥ ngưỡng, một-một, tham lam theo IoU giảm dần).

Kết quả "hit table": các cặp (tid, frame) mà box GT của track tid ở khung frame được ghép với ≥1 detection
(một GT ghép nhiều nhất 1 detection và ngược lại). Ghép trên MỌI box GT xe (kể cả không visible) để detection
của xe bị che không bị gán nhầm; KPI sau đó chỉ xét khung thuộc cửa sổ KPI của sự kiện.
Cache: paths.HITS_DIR/<dump_name>/<dataset>_<seq>.parquet (tid, frame).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from paths import HITS_DIR as HITS  # noqa: E402

IOU_THR = 0.5


def iou_matrix(a, b):
    """a: (n,4) x,y,w,h ; b: (m,4) → (n,m) IoU."""
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou, thr=IOU_THR):
    """Trả mảng bool cho hàng (GT) được ghép. Tham lam theo IoU giảm dần, một-một."""
    n, m = iou.shape
    matched = np.zeros(n, bool)
    if n == 0 or m == 0:
        return matched
    ii, jj = np.nonzero(iou >= thr)
    if len(ii) == 0:
        return matched
    order = np.argsort(-iou[ii, jj], kind="stable")
    used_d = np.zeros(m, bool)
    for k in order:
        i, j = ii[k], jj[k]
        if not matched[i] and not used_d[j]:
            matched[i] = True
            used_d[j] = True
    return matched


def match_seq(gt, det, thr=IOU_THR):
    """gt: DataFrame frame, tid, x,y,w,h ; det: dump DataFrame. Trả DataFrame (tid, frame) được ghép."""
    gt = gt.sort_values("frame", kind="stable")
    det = det.sort_values("frame", kind="stable")
    gf, df_ = gt.frame.values, det.frame.values
    gb, db = gt[["x", "y", "w", "h"]].values.astype(float), det[["x", "y", "w", "h"]].values.astype(float)
    gtid = gt.tid.values
    out_t, out_f = [], []
    frames = np.unique(gf)
    gs, ge = np.searchsorted(gf, frames, "left"), np.searchsorted(gf, frames, "right")
    ds, de = np.searchsorted(df_, frames, "left"), np.searchsorted(df_, frames, "right")
    for f, a0, a1, b0, b1 in zip(frames, gs, ge, ds, de):
        if b1 <= b0:
            continue
        m = greedy_match(iou_matrix(gb[a0:a1], db[b0:b1]), thr)
        if m.any():
            out_t.append(gtid[a0:a1][m])
            out_f.append(np.full(int(m.sum()), f))
    if not out_t:
        return pd.DataFrame(dict(tid=np.array([], int), frame=np.array([], int)))
    return pd.DataFrame(dict(tid=np.concatenate(out_t), frame=np.concatenate(out_f)))


def hits_for_dump(dump_dir, gt_rows, dataset="uavdt", seqs=None, thr=IOU_THR, cache=True):
    """dict seq → DataFrame(tid, frame). gt_rows: rows parquet (seq, frame, tid, x, y, w, h)."""
    out = {}
    cdir = HITS / Path(dump_dir).name
    for seq, g in gt_rows.groupby("seq"):
        if seqs is not None and seq not in seqs:
            continue
        cp = cdir / f"{dataset}_{seq}.parquet"
        if cache and cp.exists():
            out[seq] = pd.read_parquet(cp)
            continue
        det = pd.read_parquet(Path(dump_dir) / f"{dataset}_{seq}.parquet")
        h = match_seq(g, det, thr)
        if cache:
            cdir.mkdir(parents=True, exist_ok=True)
            h.to_parquet(cp, index=False)
        out[seq] = h
    return out
