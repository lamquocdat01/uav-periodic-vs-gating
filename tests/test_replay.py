"""P2A-A8: unit test engine replay trên dữ liệu TỔNG HỢP (L7). Chạy: python -m pytest tests/test_replay.py
(hoặc python tests/test_replay.py nếu chưa có pytest)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
from p2_dump_schema import DumpError, validate  # noqa: E402
from p2_engine import (BIG, EventSet, Timeline, activation, gate_or_refresh, kpis, matched_periodic_S,  # noqa: E402
                       onset_mask, onset_oracle, periodic)
from p2_match import match_seq  # noqa: E402


def lemma_miss(D, S, r):
    x = D / S
    n = math.floor(x + 1e-12)
    f = x - n
    return (1 - r) ** n * (1 - f * r)


def synth(D_list, L=400, gap=60):
    """Một chuỗi dài L với các sự kiện (track i) bắt đầu cách nhau, hit ở mọi khung (r=1)."""
    starts = [10 + i * gap for i in range(len(D_list))]
    ev = pd.DataFrame(dict(seq="S1", track_id=np.arange(len(D_list)), start=starts, D=D_list))
    hits = pd.DataFrame(dict(tid=np.concatenate([np.full(D, i) for i, D in enumerate(D_list)]),
                             frame=np.concatenate([np.arange(s, s + D) for s, D in zip(starts, D_list)])))
    tl = Timeline({"S1": L})
    return tl, ev, {"S1": hits}


def test_periodic_lemma1_r1():
    D_list = [1, 2, 3, 5, 7, 8, 12, 19, 25, 40]
    tl, ev, hits = synth(D_list, L=40 + 60 * len(D_list))
    es = EventSet(ev, hits, tl)
    for S in (1, 2, 3, 4, 5, 7, 10, 16, 20, 37):
        det = kpis(es.first_latency(periodic(tl, S)))["det"]
        want = np.array([1 - lemma_miss(D, S, 1.0) for D in D_list])
        assert np.max(np.abs(det - want)) < 1e-9, (S, det, want)


def test_periodic_latency_formula():
    D_list = [2, 4, 7, 12, 40]
    tl, ev, hits = synth(D_list, L=400)
    es = EventSet(ev, hits, tl)
    for S in (1, 3, 4, 6, 11):
        k = kpis(es.first_latency(periodic(tl, S)), (3, 5, 10))
        for Lm in (3, 5, 10):
            want = np.array([1 - lemma_miss(min(D, Lm + 1), S, 1.0) for D in D_list])
            assert np.max(np.abs(k[f"L{Lm}"] - want)) < 1e-9


def test_gate_onset_zero_latency():
    D_list = [1, 3, 9, 30]
    tl, ev, hits = synth(D_list, L=300)
    es = EventSet(ev, hits, tl)
    run = gate_or_refresh(onset_mask(tl, ev[["seq", "start"]], w=1), tl, R=math.inf)  # q_in=1, q_out=0, w=1
    lat = es.first_latency(run)
    assert np.all(lat == 0)
    assert np.all(kpis(lat, (3,))["L3"] == 1.0)
    run2 = onset_oracle(tl, ev[["seq", "start"]], q_out=0.0, seed=1)
    assert np.all(es.first_latency(run2) == 0)


def test_matched_cost_bisection():
    tl = Timeline({"A": 1000, "B": 777, "C": 1500})
    for a in (0.03, 0.07, 0.123, 0.4):
        S, aP = matched_periodic_S(tl, a)
        assert abs(aP - a) < 0.002, (a, S, aP)


def test_roi_m0_equals_unfiltered():
    # ROI m=0: vùng = toàn ảnh → đoạn trong ROI = đoạn visible → cùng KPI
    import p2_engine
    ev = p2_engine.load_events("train")
    e3, r0 = ev["E3"], ev["E3ROI0"]
    a = e3.set_index(["seq", "track_id"])[["start", "D"]].sort_index()
    b = r0.set_index(["seq", "track_id"])[["start", "D"]].sort_index()
    assert a.equals(b)


def test_dump_schema_catches_missing():
    df = pd.DataFrame(dict(frame=[1], x=[0.0], y=[0.0], w=[1.0], h=[1.0], score=[0.5]))
    try:
        validate(df)
    except DumpError as e:
        assert "cls" in str(e)
    else:
        raise AssertionError("không bắt được cột thiếu")
    ok = df.assign(cls=np.array([0], int))
    assert validate(ok)
    bad = ok.assign(score=[1.5])
    try:
        validate(bad)
    except DumpError:
        pass
    else:
        raise AssertionError("không bắt được score ngoài [0,1]")


def test_match_iou_one_to_one():
    gt = pd.DataFrame(dict(frame=[1, 1, 2], tid=[7, 8, 7], x=[0, 100, 0.0], y=[0, 0, 0.0], w=[10, 10, 10.0], h=[10, 10, 10.0]))
    det = pd.DataFrame(dict(frame=[1, 1, 2], x=[1, 1, 30.0], y=[0, 0, 0.0], w=[10, 10, 10.0], h=[10, 10, 10.0], score=0.9, cls=0))
    h = match_seq(gt, det)
    # khung 1: hai detection chồng lên track 7, chỉ một được ghép; track 8 không; khung 2: IoU = 0
    assert set(map(tuple, h[["tid", "frame"]].values)) == {(7, 1)}


def test_activation_counts():
    tl = Timeline({"A": 100})
    run = periodic(tl, 10)
    assert np.allclose(activation(run), 0.1)
    assert BIG > 10 ** 6


def test_sim_detector_recall_and_fp():
    from p2_sim_detector import simulate_seq
    rng = np.random.default_rng(3)
    n = 20000
    rows = pd.DataFrame(dict(frame=rng.integers(1, 2001, n), x=rng.uniform(0, 900, n), y=rng.uniform(0, 400, n),
                             w=rng.uniform(10, 60, n), h=rng.uniform(10, 60, n), visible=rng.random(n) < 0.9))
    nv = int(rows.visible.sum())
    d1 = simulate_seq(rows, 2000, 1024, 540, 1.0, 0.0, "T1")
    assert len(d1) == nv and (d1.cls == 0).all()
    d5 = simulate_seq(rows, 2000, 1024, 540, 0.5, 0.0, "T1")
    assert abs(len(d5) / nv - 0.5) < 4 * np.sqrt(0.25 / nv)
    df = simulate_seq(rows, 2000, 1024, 540, 1.0, 0.05, "T1")
    n_fp = int((df.cls == -1).sum())
    assert abs(n_fp - 100) < 4 * np.sqrt(100)
    assert (df.x >= 0).all() and (df.x + df.w <= 1024 + 1e-6).all()


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
