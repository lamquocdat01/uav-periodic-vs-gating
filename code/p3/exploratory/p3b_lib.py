"""P3b [POST HOC — không thuộc họ xác nhận]: hàm thuần (kiểm được bằng dữ liệu tổng hợp) cho các phân tích khám phá E1–E5.

Không sửa code/p2; chỉ import engine replay đóng băng. Không ghi đè kết quả P3 (results/p3/score_test.json, cells_test.csv, RESULTS_28.md).
"""
import math

import numpy as np
from scipy.stats import spearmanr

POST_HOC = "[POST HOC — không thuộc họ xác nhận]"


def recall_by_k(ev, hit_set, vis_set, kmax=10):
    """ev: DataFrame(seq, track_id, start, D, e3_type); hit_set / vis_set: set (seq, tid, frame).
    Trả list dòng (seq, e3_type, k, n, hit) — khung start+k với k < D và box GT VISIBLE tồn tại."""
    out = []
    for e in ev.itertuples(index=False):
        for k in range(min(kmax, int(e.D) - 1) + 1):
            key = (e.seq, int(e.track_id), int(e.start) + k)
            if key in vis_set:
                out.append((e.seq, e.e3_type, k, 1, int(key in hit_set)))
    return out


def pooled_boot(df, by, B=1000, seed=42):
    """df: cột seq, n, hit (+ by). Trả {nhóm: (r, lo, hi, n)} — bootstrap theo chuỗi."""
    rng = np.random.default_rng(seed)
    seqs = sorted(df.seq.unique())
    idx = rng.integers(0, len(seqs), (B, len(seqs)))
    Wt = np.stack([np.bincount(i, minlength=len(seqs)) for i in idx])
    res = {}
    for key, g in df.groupby(by):
        s = g.groupby("seq")[["n", "hit"]].sum().reindex(seqs, fill_value=0)
        n, h = s.n.values.astype(float), s.hit.values.astype(float)
        bs = (Wt @ h) / np.maximum(Wt @ n, 1e-12)
        res[key] = (float(h.sum() / max(n.sum(), 1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(n.sum()))
    return res


def window_mask(n_frames, onsets, w):
    """{seq: bool[L]}: khung trong ≥ 1 cửa sổ [onset, onset + w) (onset 1-based)."""
    out = {}
    for s, L in n_frames.items():
        m = np.zeros(L, bool)
        for o in onsets.get(s, []):
            m[int(o) - 1: min(int(o) - 1 + w, L)] = True
        out[s] = m
    return out


def q_in_out(fire_by_seq, mask_by_seq, seqs=None):
    """q_in = P(bắn | trong cửa sổ), q_out = P(bắn | ngoài) gộp các chuỗi."""
    seqs = seqs if seqs is not None else sorted(fire_by_seq)
    kin = nin = kout = nout = 0
    for s in seqs:
        f, m = fire_by_seq[s], mask_by_seq[s]
        kin += f[m].sum(); nin += m.sum(); kout += f[~m].sum(); nout += (~m).sum()
    return float(kin / max(nin, 1)), float(kout / max(nout, 1)), int(nin), int(nout)


def calibration(pred, obs):
    """OLS obs = a + b·pred; MAE; Spearman; bias = mean(obs − pred)."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    b, a = np.polyfit(pred, obs, 1)
    rho, p = spearmanr(pred, obs)
    return dict(n=int(len(pred)), slope=float(b), intercept=float(a), mae=float(np.mean(np.abs(obs - pred))),
                bias=float(np.mean(obs - pred)), spearman=float(rho), spearman_p=float(p))


def first_cross(x, adv):
    """điểm adv đổi dấu + → ≤ 0 (nội suy tuyến tính); +inf nếu luôn > 0; 0 nếu ngay đầu ≤ 0 (như p2_mstar_iso)."""
    x, adv = np.asarray(x, float), np.asarray(adv, float)
    for i in range(len(x) - 1):
        if adv[i] > 0 and adv[i + 1] <= 0:
            return float(x[i] + (x[i + 1] - x[i]) * adv[i] / (adv[i] - adv[i + 1]))
    return math.inf if adv[0] > 0 else 0.0


def qo_star_pred(eps, T, rho1, M, q_b=0.0, c=0.0):
    """Thm P3-iii / Cor. G5 (r = 1): gate rẻ hơn ở iso-KPI iff ρ1 + (1−ρ1) q_out,eff(M) + c < (1−ε)/T,
    q_out,eff = 1 − (1−q_b)(1−q_o)^M ⇒ q_o* = 1 − ((1 − q*)/(1 − q_b))^(1/M), q* = ((1−ε)/T − ρ1 − c)/(1 − ρ1).
    0 nếu q* ≤ q_b (gate không bao giờ rẻ hơn)."""
    qs = ((1 - eps) / T - rho1 - c) / (1 - rho1)
    if qs <= q_b:
        return 0.0
    return float(1 - ((1 - qs) / (1 - q_b)) ** (1.0 / M))
