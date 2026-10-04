"""P2A-B4 test: dump OpenVINO trên 20 ảnh tổng hợp với trọng số COCO tạm (yolo26n@320 export vào paths.TEST_MODELS_DIR).
Bỏ qua (SKIP) nếu chưa export IR. Kiểm: định dạng A1 hợp lệ, đủ khung, t_ms > 0; cue tiny_det trả score ∈ [0,1]."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
sys.path.insert(0, str(ROOT / "code"))
from paths import FIXTURES, TEST_MODELS_DIR  # noqa: E402

IR = TEST_MODELS_DIR / "yolo26n_openvino_model"
IMGS = FIXTURES / "synth_imgs"


def _imgs():
    from p2_cues import synth_texture
    IMGS.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(20):
        p = IMGS / f"img{i + 1:06d}.jpg"
        if not p.exists():
            im = synth_texture(seed=100 + i)
            cv2.rectangle(im, (200 + 10 * i, 200), (320 + 10 * i, 280), (30, 30, 200), -1)  # vật thể màu
            cv2.imwrite(str(p), im)
        paths.append(p)
    return paths


def test_dump_schema_on_synthetic():
    if not IR.exists():
        print("SKIP (chưa có IR)")
        return
    from p2_dump import OVDetector, dump_images
    from p2_dump_schema import validate
    paths = _imgs()
    det = OVDetector(IR, 320, "CPU", conf_min=0.01)
    df, ms = dump_images(det, paths)
    df["cls"] = df.cls.astype(int)
    df["frame"] = df.frame.astype(int)
    assert validate(df, len(paths))
    assert len(ms) == 20 and min(ms) > 0


def test_drop_degenerate():
    """P2E: box w/h ≤ 0 bị bỏ (schema cấm), box hợp lệ giữ nguyên, trả số box bỏ."""
    import pandas as pd
    from p2_dump import drop_degenerate
    from p2_dump_schema import validate
    df = pd.DataFrame(dict(frame=[1, 1, 2, 2], x=[0.0, 5, 10, 1023], y=[0.0, 5, 3, 7], w=[4.0, 0, 2, 3], h=[3.0, 2, -1, 0.5],
                           score=[0.9, 0.3, 0.2, 0.06], cls=[3, 3, 4, 5]))
    out, n = drop_degenerate(df)
    assert n == 2 and list(out.x) == [0.0, 1023] and validate(out, 2)


def test_tiny_det_cue():
    if not IR.exists():
        print("SKIP (chưa có IR)")
        return
    from p2_cues import TinyDet
    paths = _imgs()
    cue = TinyDet(IR, 320, "CPU")
    s, t = cue(None, cv2.imread(str(paths[0])))
    assert 0.0 <= s <= 1.0 and t > 0


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
