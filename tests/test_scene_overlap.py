"""P2A-B5 test (tổng hợp): ảnh nén lại JPEG + đổi sáng nhẹ → Hamming ≤ 6; ảnh cảnh khác → > 6; run() gắn cờ đúng chuỗi."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_cues import synth_texture  # noqa: E402
from p2_scene_overlap import hamming_matrix, phash, run  # noqa: E402

TMP = ROOT / "tests" / "fixtures" / "overlap"


def test_phash_near_and_far():
    a = synth_texture(seed=11)
    ok, enc = cv2.imencode(".jpg", cv2.convertScaleAbs(a, alpha=1.05, beta=5), [cv2.IMWRITE_JPEG_QUALITY, 60])
    b = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    c = synth_texture(seed=12)
    H = np.stack([phash(a), phash(b), phash(c)])
    d = hamming_matrix(H, H)
    assert d[0, 1] <= 6, d
    assert d[0, 2] > 6, d


def test_run_flags_duplicate_sequence():
    det = TMP / "det"
    det.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        cv2.imwrite(str(det / f"00000{i}_00001_d_0000001.jpg"), synth_texture(seed=200 + i))
    seqA = [TMP / "A" / f"img{i:06d}.jpg" for i in range(1, 4)]
    seqB = [TMP / "B" / f"img{i:06d}.jpg" for i in range(1, 4)]
    for p in seqA + seqB:
        p.parent.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(seqA):  # chuỗi A dùng lại cảnh DET số 2
        cv2.imwrite(str(p), cv2.convertScaleAbs(synth_texture(seed=202), alpha=1.0, beta=3 * i))
    for i, p in enumerate(seqB):
        cv2.imwrite(str(p), synth_texture(seed=300 + i))
    out = run(det, [("x", seqA, "test"), ("x", seqB, "test")])
    rA, rB = out["rows"]
    assert rA["n_near"] >= 1 and rB["n_near"] == 0, (rA, rB)
    assert out["share_seq_near_dup"] == 0.5


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
