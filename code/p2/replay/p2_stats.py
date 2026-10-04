"""P2A-A6: thống kê cho replay — paired bootstrap theo chuỗi (B=1000, seed=42), Holm, điểm giao có CI.

p_boot = p hai phía xấp xỉ chuẩn từ SE bootstrap (liên tục); p_perc = p theo phân vị (≥ 1/B).
Đầu vào chuẩn: vector theo sự kiện (xác suất bỏ lỡ, đã trung bình theo pha) cho hai chính sách trên CÙNG tập sự kiện,
và chỉ số chuỗi của từng sự kiện. Δ = miss_P − miss_G (> 0: gate thắng).
"""
import numpy as np
from scipy.stats import norm

B, SEED = 1000, 42


def boot_weights(n_seq, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    W = np.zeros((B, n_seq))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n_seq)
    return W


def seq_sums(values, seq_idx, n_seq):
    return np.bincount(seq_idx, weights=values, minlength=n_seq), np.bincount(seq_idx, minlength=n_seq).astype(float)


def boot_mean(values, seq_idx, n_seq, W):
    s, n = seq_sums(values, seq_idx, n_seq)
    point = s.sum() / max(n.sum(), 1e-12)
    bs = (W @ s) / np.maximum(W @ n, 1e-12)
    return float(point), bs


def paired_delta(miss_P, miss_G, seq_idx, n_seq, W):
    """Δ = mean(miss_P) − mean(miss_G), bootstrap theo chuỗi với CÙNG trọng số. Trả dict điểm, CI, p hai phía."""
    pt, bs = boot_mean(np.asarray(miss_P) - np.asarray(miss_G), seq_idx, n_seq, W)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_perc = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    # p liên tục (xấp xỉ chuẩn với SE bootstrap): p theo phân vị bị chặn dưới ở 1/B nên Holm trên nhiều ô không bao giờ bác bỏ
    se = float(np.std(bs, ddof=1))
    p_norm = float(2 * norm.sf(abs(pt) / se)) if se > 0 else (0.0 if pt != 0 else 1.0)
    return dict(delta=pt, lo=float(lo), hi=float(hi), se_boot=se, p_boot=p_norm, p_perc=float(min(1.0, max(p_perc, 1.0 / len(bs)))),
                sign=("+" if lo > 0 else "-" if hi < 0 else "0"))


def holm(pvals, alpha=0.05):
    """Holm–Bonferroni. Trả (p_adj, reject)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj, adj <= alpha


def crossing(x, y_boot, y_point):
    """Điểm giao x* nơi y đổi dấu (+ → −) theo nội suy tuyến tính; CI từ bootstrap. y_boot: (B × len(x))."""
    def cross1(y):
        s = np.sign(y)
        for i in range(len(x) - 1):
            if s[i] > 0 and s[i + 1] <= 0:
                return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
        return np.nan if s[0] <= 0 else np.inf
    pt = cross1(np.asarray(y_point))
    bs = np.array([cross1(y) for y in y_boot])
    fin = bs[np.isfinite(bs)]
    ci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.5 * len(bs) else None
    return dict(x_star=float(pt), ci=ci, share_inf=float(np.isinf(bs).mean()), share_nan=float(np.isnan(bs).mean()))
