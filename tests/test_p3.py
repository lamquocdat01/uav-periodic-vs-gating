"""P3 tests: mã hoá luật cỡ mẫu TEST → CHƯA KẾT LUẬN; bootstrap a_G − a_P(ε) trên lưới; quan sát H8 ghép cặp; khâu dựng ô TRAIN."""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
sys.path.insert(0, str(ROOT / "code" / "p3"))
from p2_prereg_score import judge, judge_iso, score  # noqa: E402
from p2_stats import boot_weights, paired_delta  # noqa: E402
import p3_replay_test as R  # noqa: E402


def test_size_rule_encoding_is_inconclusive():
    for pred in ("+", "<=0"):
        assert judge(pred, R.INF_CI[0], R.INF_CI[1]) == "inconclusive"
    assert judge_iso("<=0", dict(gate_cheaper=False, diff_lo=-math.inf, diff_hi=math.inf)) == "inconclusive"


def test_score_keeps_family_with_inf_ci():
    cells = [dict(id="a", confirmatory=True, sign_pred="+", comparison="matched_cost", hypothesis="H3/H5"),
             dict(id="b", confirmatory=True, sign_pred="<=0", comparison="iso_kpi", hypothesis="H2")]
    obs = [dict(id="a", delta=float("nan"), lo=-math.inf, hi=math.inf, p_boot=None),
           dict(id="b", gate_cheaper=False, diff_lo=-math.inf, diff_hi=math.inf)]
    s = score(cells, obs)
    assert s["family_size"] == 2 and s["n_scored"] == 2 and s["three_way"]["inconclusive"] == 1.0 and s["verdict"] == "untestable"


def test_boot_iso_diff_identity_weights():
    """W = 1 (không lấy mẫu lại) → a_G − a_P bằng tính tay trên lưới."""
    nseq, nS = 3, len(R.S_GRID)
    W = np.ones((1, nseq))
    frames = np.array([100.0, 200.0, 300.0])
    runsG = np.array([20.0, 30.0, 10.0])
    act = np.outer(1 / R.S_GRID, frames)                 # periodic: frames / S chạy mỗi chuỗi
    miss = np.outer(np.linspace(0, 0.5, nS), [10, 10, 10])  # miss tăng theo S
    nev = np.array([10.0, 10.0, 10.0])
    eps = 0.02
    d = R.boot_iso_diff(W, frames, runsG, 0.01, act, miss, nev, eps)
    j = np.flatnonzero(np.linspace(0, 0.5, nS) <= eps + 1e-12).max()
    want = 60 / 600 + 0.01 - 1 / R.S_GRID[j]
    assert abs(d[0] - want) < 1e-12


def test_boot_iso_diff_infeasible():
    W = np.ones((2, 2))
    d = R.boot_iso_diff(W, np.array([10.0, 10.0]), np.array([1.0, 1.0]), 0.0, np.ones((len(R.S_GRID), 2)),
                        np.full((len(R.S_GRID), 2), 5.0), np.array([5.0, 5.0]), 0.02)
    assert np.all(np.isneginf(d))


def test_h8_paired_delta_equals_difference():
    rng = np.random.default_rng(1)
    n, nseq = 120, 6
    sidx = rng.integers(0, nseq, n)
    W = boot_weights(nseq)
    dlo, dhi = rng.random(n), rng.random(n)
    d = paired_delta(dlo, dhi, sidx, nseq, W)
    assert abs(d["delta"] - (dlo.mean() - dhi.mean())) < 1e-12


def test_train_cell_construction_matches_frozen_counts():
    """Khâu dựng ô của p3_replay_test (load_events + ego ngưỡng đóng băng) cho đúng n_events_train / n_seq_train đóng băng."""
    import json
    import pandas as pd
    from p2_engine import load_events
    cells = json.loads(R.CELLS.read_text(encoding="utf-8"))["cells"]
    ego, _ = R.ego_map("train")
    E3 = load_events("train")["E3"]
    E3["ego"] = E3.seq.map(ego)
    for c in cells:
        if c["confirmatory"]:
            sub = E3[(E3.ego == c["ego"]) & (E3.M_bin == c["M_bin"])]
            assert (len(sub), sub.seq.nunique()) == (c["n_events_train"], c["n_seq_train"]), c["id"]


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
