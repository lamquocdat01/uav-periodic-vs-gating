"""P2A-B3 test: kênh tổng hợp với q_in, q_b, q_o biết trước → ước lượng khôi phục được; split ≠ train → AssertionError (L8)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_estimate_channel import estimate, fit_qb_qo, in_window_mask  # noqa: E402

Q_IN, Q_B, Q_O, W = 0.8, 0.02, 0.01, 5


def synth(n_seq=12, L=3000, lam=0.02, seed=7):
    rng = np.random.default_rng(seed)
    traces, n_frames, onsets, M = {}, {}, {}, {}
    for i in range(n_seq):
        s = f"S{i:02d}"
        n_frames[s] = L
        ons = np.flatnonzero(rng.random(L) < lam) + 1
        onsets[s] = list(ons)
        Mv = np.repeat(rng.integers(0, 45, L // 100 + 1), 100)[:L]
        M[s] = Mv
        inw = in_window_mask({s: L}, {s: ons}, W, "train")[s]
        p = np.where(inw, Q_IN, 1 - (1 - Q_B) * (1 - Q_O) ** Mv)
        fire = rng.random(L) < p
        score = np.where(fire, rng.uniform(0.5, 1, L), rng.uniform(0, 0.5, L))
        traces[s] = pd.DataFrame(dict(frame=np.arange(1, L + 1), cue="synth", score=score, t_ms=1.0))
    return traces, n_frames, onsets, M


def test_recover_channel():
    traces, n_frames, onsets, M = synth()
    out = estimate(traces, n_frames, onsets, M, "train", t_det_ms=100.0, theta_grid=[0.5], B_=200)
    op = [o for o in out["operating_points"] if o["w"] == W][0]
    assert abs(op["q_in"] - Q_IN) < 0.03, op["q_in"]
    assert op["qbqo_ci"]["q_o"][0] <= Q_O <= op["qbqo_ci"]["q_o"][1], (op["q_o"], op["qbqo_ci"])
    assert op["qbqo_ci"]["q_b"][0] <= Q_B <= op["qbqo_ci"]["q_b"][1], (op["q_b"], op["qbqo_ci"])
    assert abs(op["c"] - 0.01) < 1e-9


def test_fit_direct():
    rng = np.random.default_rng(1)
    Mv = rng.integers(0, 60, 200_000)
    fire = rng.random(len(Mv)) < 1 - (1 - 0.05) * (1 - 0.02) ** Mv
    qb, qo = fit_qb_qo(Mv, fire, "train")
    assert abs(qb - 0.05) < 0.005 and abs(qo - 0.02) < 0.002, (qb, qo)


def test_test_split_forbidden():
    traces, n_frames, onsets, M = synth(n_seq=3, L=500)
    for fn, args in ((estimate, (traces, n_frames, onsets, M, "test")), (fit_qb_qo, (np.zeros(3), np.zeros(3, bool), "test")),
                     (in_window_mask, (n_frames, onsets, 3, "test"))):
        try:
            fn(*args)
        except AssertionError:
            continue
        raise AssertionError(f"{fn.__name__} chấp nhận split=test")


def test_ego_labels_and_c_real(tmp=ROOT / "tests" / "fixtures" / "p2e_channel"):
    """P2E: nhãn ego theo ngưỡng đề xuất từ phân bố (unimodal → không tách); c từ bench_c_real (TEST → AssertionError)."""
    import json
    from p2_estimate_channel import c_from_real, ego_labels
    tmp.mkdir(parents=True, exist_ok=True)
    eg = pd.DataFrame(dict(dataset=["UAVDT"] * 3 + ["VisDrone"], seq=["A", "B", "C", "V"], shift_px_median=[0.3, 2.0, 9.0, 0.1]))
    hj = tmp / "hover.json"
    hj.write_text(json.dumps({"by_dataset": {"UAVDT": {"proposed_threshold_px": 1.0}}}), encoding="utf-8")
    lab, thr = ego_labels(eg, hj)
    assert thr == 1.0 and lab == {"A": "hovering", "B": "moving", "C": "moving"}, lab
    hj.write_text(json.dumps({"by_dataset": {"UAVDT": {"proposed_threshold_px": None}}}), encoding="utf-8")
    assert ego_labels(eg, hj) == ({}, None)
    cj = tmp / "c.json"
    cj.write_text(json.dumps({"split": "train", "denominator": "yolo26s_1024/GPU",
                              "c": {"raw_diff/1thr": {"c_median": 0.01}, "tiny_det_yolo26n_320/CPU": {"c_median": 0.2}}}), encoding="utf-8")
    c, den = c_from_real(cj)
    assert c == {"raw_diff": 0.01, "tiny_det": 0.2} and den == "yolo26s_1024/GPU"
    traces, n_frames, onsets, M = synth(n_seq=4, L=600)
    ch = estimate(traces, n_frames, onsets, M, "train", t_det_ms=100.0, B_=20, c_by_cue={"synth": 0.123})
    assert all(op["c"] == 0.123 for op in ch["operating_points"])
    cj.write_text(json.dumps({"split": "test", "denominator": "x", "c": {}}), encoding="utf-8")
    try:
        c_from_real(cj)
    except AssertionError:
        return
    raise AssertionError("c_from_real chấp nhận split=test")


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
