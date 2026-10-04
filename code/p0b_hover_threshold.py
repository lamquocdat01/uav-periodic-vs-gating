"""P0b-D: đề xuất ngưỡng hovering TỪ PHÂN BỐ (không áp cứng 1.0 px/khung).

Đầu vào: results/p0/egomotion_seq.parquet (p0_egomotion.py; cần frames). Biến: x = log10(dịch tâm ảnh qua H,
median theo chuỗi, px/khung). Hai cách, báo cả hai:
  (a) khe lớn nhất giữa hai giá trị liên tiếp (mỗi phía ≥ 3 chuỗi) → ngưỡng = trung điểm hình học của khe;
  (b) hỗn hợp 2 Gauss 1-D trên x (EM tự viết, seed=42) → ngưỡng = điểm posterior 0.5 giữa hai mode.
Không có hai mode rõ (khe nhỏ, hai thành phần chồng lấn) → ghi "unimodal" và KHÔNG đề xuất.
Xuất results/p0/hover_threshold.json + histogram theo dataset (bin log) trong json.
Chạy: python code/p0b_hover_threshold.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results" / "p0"
OUT = P0 / "hover_threshold.json"
MIN_SIDE, SEED = 3, 42


def largest_gap(x):
    x = np.sort(x)
    if len(x) < 2 * MIN_SIDE:
        return None
    gaps = np.diff(x)
    idx = np.arange(MIN_SIDE - 1, len(x) - MIN_SIDE)
    i = idx[np.argmax(gaps[idx])]
    return dict(lo=float(10 ** x[i]), hi=float(10 ** x[i + 1]), gap_log10=float(gaps[i]),
                threshold_px=float(10 ** ((x[i] + x[i + 1]) / 2)), n_below=int(i + 1), n_above=int(len(x) - i - 1))


def gmm2(x, iters=500):
    rng = np.random.default_rng(SEED)
    mu = np.percentile(x, [25, 75]).astype(float) + rng.normal(0, 1e-3, 2)
    sd = np.full(2, x.std() / 2 + 1e-3)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
        g = pdf / np.maximum(pdf.sum(1, keepdims=True), 1e-300)
        nk = g.sum(0)
        w, mu = nk / len(x), (g * x[:, None]).sum(0) / nk
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-3
    o = np.argsort(mu)
    mu, sd, w = mu[o], sd[o], w[o]
    grid = np.linspace(mu[0], mu[1], 2001)
    p = [w[k] / sd[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) for k in (0, 1)]
    cross = grid[np.argmin(np.abs(p[0] - p[1]))]
    ashman_d = float(np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))  # D > 2: tách mode rõ
    return dict(mu_px=[float(10 ** m) for m in mu], sd_log10=[float(s) for s in sd], weight=[float(v) for v in w],
                threshold_px=float(10 ** cross), ashman_D=ashman_d, bimodal=bool(ashman_d > 2))


def main():
    p = P0 / "egomotion_seq.parquet"
    if not p.exists():
        OUT.write_text(json.dumps({"status": "NO_FRAMES", "note": "cần frames → chạy p0_egomotion.py trước"}, indent=1), encoding="utf-8")
        print("chưa có egomotion_seq.parquet (chưa có frames) → ghi NO_FRAMES")
        return
    s = pd.read_parquet(p)
    out = {"status": "OK", "by_dataset": {}}
    for ds, d in list(s.groupby("dataset")) + [("ALL", s)]:
        v = d.shift_px_median.dropna().values
        v = v[v > 0]
        x = np.log10(v)
        hist, edges = np.histogram(x, bins=np.arange(np.floor(x.min() * 4) / 4, np.ceil(x.max() * 4) / 4 + 0.25, 0.25))
        gap, gm = largest_gap(x), gmm2(x) if len(x) >= 2 * MIN_SIDE else None
        prop = None
        if gm and gm["bimodal"]:
            prop = gm["threshold_px"]
        elif gap and gap["gap_log10"] >= 0.3:
            prop = gap["threshold_px"]
        out["by_dataset"][ds] = dict(n_seq=int(len(v)), quantiles_px={f"p{q}": float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)},
                                     hist_log10_edges=[float(e) for e in edges], hist_counts=[int(c) for c in hist],
                                     largest_gap=gap, gmm2=gm, proposed_threshold_px=prop,
                                     n_hover_at_proposed=(int((v < prop).sum()) if prop else None),
                                     verdict=("bimodal" if prop else "unimodal — không đề xuất ngưỡng từ dữ liệu"))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["proposed_threshold_px"], v["verdict"]) for k, v in out["by_dataset"].items()}, indent=1))


if __name__ == "__main__":
    main()
