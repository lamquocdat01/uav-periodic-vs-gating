"""P3b [POST HOC] tests trên dữ liệu tổng hợp: r(k), bootstrap gộp, q_in/q_out, hiệu chuẩn, điểm giao q_o*, công thức Thm P3-iii."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p3" / "exploratory"))
import p3b_lib as lib  # noqa: E402


def test_recall_by_k_counts_visible_frames_within_D():
    ev = pd.DataFrame(dict(seq=["A"], track_id=[1], start=[10], D=[3], e3_type=["border"]))
    vis = {("A", 1, 10), ("A", 1, 11), ("A", 1, 12), ("A", 1, 13)}
    hit = {("A", 1, 11)}
    rows = lib.recall_by_k(ev, hit, vis, kmax=10)
    assert [(r[2], r[4]) for r in rows] == [(0, 0), (1, 1), (2, 0)]  # k = 3 ngoài D


def test_pooled_boot_point_and_ci():
    df = pd.DataFrame(dict(seq=["a", "a", "b", "b"], k=[0, 0, 0, 0], n=[1, 1, 1, 1], hit=[1, 1, 0, 0]))
    r = lib.pooled_boot(df, "k", B=200)[0]
    assert r[0] == 0.5 and r[3] == 4 and 0.0 <= r[1] <= 0.5 <= r[2] <= 1.0


def test_window_mask_and_q_in_out():
    m = lib.window_mask({"s": 10}, {"s": [3]}, 2)
    assert m["s"].tolist() == [False, False, True, True] + [False] * 6
    fire = {"s": np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0], bool)}
    qi, qo, nin, nout = lib.q_in_out(fire, m)
    assert (qi, nin, nout) == (0.5, 2, 8) and abs(qo - 1 / 8) < 1e-12


def test_calibration_exact_line():
    x = np.array([-0.3, -0.1, 0.0, 0.2])
    c = lib.calibration(x, 2 * x + 0.1)
    assert abs(c["slope"] - 2) < 1e-9 and abs(c["intercept"] - 0.1) < 1e-9 and abs(c["spearman"] - 1) < 1e-12


def test_first_cross_cases():
    assert abs(lib.first_cross([0, 1, 2], [1.0, 0.5, -0.5]) - 1.5) < 1e-12
    assert lib.first_cross([0, 1], [1.0, 1.0]) == math.inf
    assert lib.first_cross([0, 1], [-1.0, -1.0]) == 0.0


def test_qo_star_pred_boundary():
    eps, T, rho1, M = 0.02, 4, 0.03, 13
    q = lib.qo_star_pred(eps, T, rho1, M)
    q_eff = 1 - (1 - q) ** M
    assert abs(rho1 + (1 - rho1) * q_eff - (1 - eps) / T) < 1e-12   # đúng trên biên
    assert lib.qo_star_pred(eps, 40, rho1, M) == 0.0               # (1−ε)/T < ρ1 → gate không bao giờ rẻ hơn


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
