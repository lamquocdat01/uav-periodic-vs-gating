"""P2A-B1 test (dữ liệu tổng hợp): dịch nền 3 px → ego_comp ≈ 0, raw_diff > 0; xe vào mép → border_band bắn;
thay đổi ở giữa ảnh → border_band không bắn; orb_lite_diff (P2E) bù được dịch nền."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p2_cues import border_band, ego_comp_diff, orb_lite_diff, raw_diff, synth_texture  # noqa: E402


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_shift_ego_vs_raw():
    a = synth_texture(seed=1)
    b = shift(a, 3, 0)
    rd, _ = raw_diff(a, b)
    meta = {}
    ec, t = ego_comp_diff(a, b, meta)
    assert meta["H_ok"]
    assert rd > 0.02, rd
    assert ec < 0.2 * rd and ec < 0.01, (ec, rd)
    assert t > 0


def test_orb_lite_compensates_shift():
    a = synth_texture(seed=1)
    b = shift(a, 6, 0)  # ≈ 1,9 px ở 320 px
    rd, _ = raw_diff(a, b)
    meta = {}
    ol, t = orb_lite_diff(a, b, meta)
    assert meta["H_ok"]
    assert ol < 0.3 * rd, (ol, rd)
    assert orb_lite_diff(None, a)[0] == 0.0 and t > 0


def test_border_entry_fires():
    a = synth_texture(seed=2)
    b = a.copy()
    cv2.rectangle(b, (0, 250), (30, 290), (255, 255, 255), -1)  # xe vào từ mép trái
    bb, _ = border_band(a, b)
    rd, _ = raw_diff(a, b)
    assert bb > 0.01 and bb > 3 * rd, (bb, rd)
    c = a.copy()
    cv2.rectangle(c, (500, 250), (540, 290), (255, 255, 255), -1)  # thay đổi ở giữa
    bc, _ = border_band(a, c)
    assert bc == 0.0, bc


def test_first_frame_zero():
    a = synth_texture(seed=3)
    assert raw_diff(None, a)[0] == 0.0 and border_band(None, a)[0] == 0.0 and ego_comp_diff(None, a)[0] == 0.0


def test_cue_trace_on_synthetic_sequence():
    import p2_cue_trace as ct
    d = ROOT / "tests" / "fixtures" / "trace_seq"
    d.mkdir(parents=True, exist_ok=True)
    base = synth_texture(seed=5)
    paths = []
    for i in range(6):
        pth = d / f"img{i + 1:06d}.jpg"
        cv2.imwrite(str(pth), shift(base, 2 * i, 0))
        paths.append(pth)
    orig = ct.frame_paths
    ct.frame_paths = lambda ds, seq, split=None: paths
    try:
        cues = ["raw_diff", "ego_comp_diff", "orb_lite_diff", "border_band"]
        df = ct.trace_seq("uavdt", "SYN", "train", cues, {c: ct.get_cue(c) for c in cues})
    finally:
        ct.frame_paths = orig
    assert len(df) == 6 * len(cues) and set(df.cue) == set(cues)
    assert (df[df.frame == 1].score == 0).all()
    assert df.score.between(0, 1).all() and (df.t_ms >= 0).all()


def test_ov_model_path_alias():
    """P2E: tên Kaggle *_openvino_<s> → junction *_openvino_model (Ultralytics nhận), cùng nội dung, gọi lại không tạo mới."""
    from p2_cues import ov_model_path
    fx = ROOT / "tests" / "fixtures" / "ov_alias"
    src = fx / "src" / "m_openvino_320"
    src.mkdir(parents=True, exist_ok=True)
    (src / "best.xml").write_text("<xml/>", encoding="utf-8")
    link = ov_model_path(src, fx / "alias")
    assert link.name == "m_openvino_320_openvino_model" and (link / "best.xml").read_text(encoding="utf-8") == "<xml/>"
    assert ov_model_path(src, fx / "alias") == link
    ok = fx / "x_openvino_model"
    assert ov_model_path(ok, fx / "alias") == ok


def test_real_ir_predict_smoke():
    """P2E: IR VisDrone thật (tên Kaggle) chạy predict được qua OVDetector / TinyDet (dry-run cũ chỉ nạp lười). SKIP nếu thiếu IR."""
    sys.path.insert(0, str(ROOT / "code"))
    from paths import MODELS_DIR, TINY_IR
    from p2_cues import TinyDet
    from p2_dump import OVDetector
    ir = MODELS_DIR / "yolo26s_visdrone_1024" / "export" / "yolo26s_visdrone_1024_openvino_640"
    if not ir.exists() or not TINY_IR.exists():
        print("SKIP test_real_ir_predict_smoke (thiếu IR)")
        return
    img = synth_texture(seed=3)
    d, ms = OVDetector(ir, 640, "CPU")(img)
    assert ms > 0 and set(d.columns) >= {"x", "y", "w", "h", "score", "cls"}
    s, t = TinyDet(TINY_IR, 320, "CPU")(None, img)
    assert 0 <= s <= 1 and t > 0


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
