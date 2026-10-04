"""P2E test: p2_bench_c_real — lấy mẫu cặp TRAIN (tất định, chia đều, f ≥ 2, split ≠ train → AssertionError) và gộp c theo lần."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_bench_c_real import ratios, sample_pairs, summarize  # noqa: E402


def _sf():
    return {f"S{i}": [f"S{i}/img{k:06d}.jpg" for k in range(1, 50 + 10 * i)] for i in range(7)} | {"E": ["E/img000001.jpg"]}


def test_sample_pairs():
    a, b = sample_pairs(_sf(), 200, "train"), sample_pairs(_sf(), 200, "train")
    assert a == b and len(a) == 200
    assert "E" not in {s for s, *_ in a}  # chuỗi 1 khung không có cặp
    counts = {s: sum(1 for x in a if x[0] == s) for s in {x[0] for x in a}}
    assert max(counts.values()) - min(counts.values()) <= 1
    for s, f, p0, p1 in a:
        assert f >= 2 and p1.endswith(f"img{f:06d}.jpg") and p0.endswith(f"img{f - 1:06d}.jpg")
    assert len({(s, f) for s, f, *_ in a}) == 200


def test_sample_pairs_test_forbidden():
    try:
        sample_pairs(_sf(), 10, "test")
    except AssertionError:
        return
    raise AssertionError("split test phải bị chặn (L8)")


def test_ratios():
    runs = [{"cue": 2.0, "det": 10.0}, {"cue": 3.0, "det": 10.0}, {"cue": 2.0, "det": 20.0}]
    c = ratios(runs, "det")["cue"]
    assert abs(c["c_median"] - 0.2) < 1e-12 and abs(c["c_min"] - 0.1) < 1e-12 and abs(c["c_max"] - 0.3) < 1e-12
    s = summarize(runs)
    assert s["det"]["median"] == 10.0 and s["cue"]["max"] == 3.0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)
