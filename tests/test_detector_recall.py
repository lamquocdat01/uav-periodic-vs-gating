"""P2E test: p2_detector_recall trên dữ liệu tổng hợp — r đúng khi biết trước box nào được phát hiện; box không visible
không vào mẫu số; ngưỡng score lọc detection; split ≠ train → AssertionError (L8)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_detector_recall import pooled_ci, recall_by_seq  # noqa: E402


def _synth(seed=0, n_f=50, n_t=4, p_det=0.7):
    rng = np.random.default_rng(seed)
    g, d = [], []
    for f in range(1, n_f + 1):
        for t in range(n_t):
            x, y = 50.0 + 100 * t, 100.0
            vis = t != 3  # track 3 luôn bị che → không vào mẫu số
            g.append(dict(frame=f, tid=t, x=x, y=y, w=40.0, h=30.0, visible=vis))
            if rng.random() < p_det:
                sc = 0.9 if rng.random() < 0.5 else 0.1
                d.append(dict(frame=f, x=x + 1, y=y + 1, w=40.0, h=30.0, score=sc, cls=3))
    return pd.DataFrame(g), pd.DataFrame(d)


def test_recall_known():
    gt, det = _synth()
    df = recall_by_seq({"S": gt}, {"S": det}, "train", 0.05)
    vis = gt[gt.visible]
    exp = len(vis.merge(det[["frame", "x"]].assign(x=lambda z: z.x - 1), on=["frame", "x"]))
    assert df.n_vis.iloc[0] == len(vis) and df.n_hit.iloc[0] == exp
    hi = recall_by_seq({"S": gt}, {"S": det}, "train", 0.25)
    assert hi.n_hit.iloc[0] < df.n_hit.iloc[0]
    r, ci = pooled_ci(pd.concat([df.assign(seq=f"S{i}") for i in range(5)]), B_=50)
    assert ci[0] <= r <= ci[1]


def test_recall_test_forbidden():
    gt, det = _synth(n_f=3)
    try:
        recall_by_seq({"S": gt}, {"S": det}, "test", 0.05)
    except AssertionError:
        return
    raise AssertionError("split test phải bị chặn (L8)")


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
