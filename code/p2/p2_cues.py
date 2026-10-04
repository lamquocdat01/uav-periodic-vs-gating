"""P2A-B1: bốn cue mức khung, giao diện chung  cue(frame_prev, frame, meta) → (score ∈ [0,1], t_ms).

  raw_diff       tỉ lệ pixel |Δ| > 25 (xám) sau khi thu nhỏ về rộng 320 px
  ego_comp_diff  ORB 2000 điểm + BF Hamming (ratio 0.75) + RANSAC 3 px → warp khung trước theo H → tỉ lệ |Δ| > 25
                 trên vùng hợp lệ (độ phân giải gốc). LÚC CHẠY KHÔNG CÓ GT → KHÔNG che bbox (khác p0_egomotion, ở đó
                 che bbox GT để đo nhiễu nền). H thất bại (< 4 cặp khớp) → trả raw diff ở độ phân giải gốc, meta_out["H_ok"]=False
  orb_lite_diff  ego_comp_diff rút gọn (P2E): ảnh thu về rộng 320 px, ORB 500 điểm, RANSAC 1 px (≈ 3 px ở 1024) — rẻ hơn
                 ego_comp_diff (ORB 1 luồng đắt hơn detector trên iGPU, xem BENCH_OPENVINO); H thất bại → raw diff 320 px
  border_band    như raw_diff nhưng chỉ đếm trong dải mép rộng 5% mỗi cạnh (ảnh 320 px)
  tiny_det       detector nhỏ (mặc định yolo26n@320, OpenVINO IR) — score = max confidence (0 nếu không có box)
Mọi cue đo t_ms (perf_counter, gồm cả tiền xử lý). frame_prev = None (khung đầu chuỗi) → score 0.
"""
import time
from pathlib import Path

import cv2
import numpy as np

DIFF_THR = 25
SMALL_W = 320
BAND = 0.05
ORB_N, RATIO, RANSAC_PX = 2000, 0.75, 3.0
ORB_LITE_N, RANSAC_PX_LITE = 500, 1.0
CUES = ["raw_diff", "ego_comp_diff", "orb_lite_diff", "border_band", "tiny_det"]


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _small(g):
    h, w = g.shape
    return cv2.resize(g, (SMALL_W, max(1, round(h * SMALL_W / w))), interpolation=cv2.INTER_AREA)


def raw_diff(prev, cur, meta=None):
    t = time.perf_counter()
    if prev is None:
        return 0.0, (time.perf_counter() - t) * 1e3
    a, b = _small(_gray(prev)), _small(_gray(cur))
    s = float((cv2.absdiff(a, b) > DIFF_THR).mean())
    return s, (time.perf_counter() - t) * 1e3


def border_band(prev, cur, meta=None):
    t = time.perf_counter()
    if prev is None:
        return 0.0, (time.perf_counter() - t) * 1e3
    a, b = _small(_gray(prev)), _small(_gray(cur))
    d = cv2.absdiff(a, b) > DIFF_THR
    h, w = d.shape
    bh, bw = max(1, round(BAND * h)), max(1, round(BAND * w))
    m = np.zeros_like(d)
    m[:bh], m[-bh:], m[:, :bw], m[:, -bw:] = True, True, True, True
    s = float(d[m].mean())
    return s, (time.perf_counter() - t) * 1e3


_ORB = {}
_BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def _comp_diff(g0, g1, n_feat, ransac_px, t, meta):
    """Bù chuyển động camera bằng homography ORB rồi đo tỉ lệ |Δ| > DIFF_THR trên vùng hợp lệ."""
    if n_feat not in _ORB:
        _ORB[n_feat] = cv2.ORB_create(nfeatures=n_feat)
    orb = _ORB[n_feat]
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    Hm = None
    if d0 is not None and d1 is not None and len(k0) >= 2 and len(k1) >= 2:
        good = [p[0] for p in _BF.knnMatch(d0, d1, k=2) if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) >= 4:
            src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    H_, W_ = g0.shape
    if Hm is None:
        s = float((cv2.absdiff(g0, g1) > DIFF_THR).mean())
        if meta is not None:
            meta["H_ok"] = False
        return s, (time.perf_counter() - t) * 1e3
    warped = cv2.warpPerspective(g0, Hm, (W_, H_))
    valid = cv2.warpPerspective(np.full_like(g0, 255), Hm, (W_, H_)) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # bỏ viền nội suy
    d = cv2.absdiff(warped, g1) > DIFF_THR
    s = float(d[valid].mean()) if valid.any() else 0.0
    if meta is not None:
        meta["H_ok"] = True
    return s, (time.perf_counter() - t) * 1e3


def ego_comp_diff(prev, cur, meta=None):
    t = time.perf_counter()
    if prev is None:
        return 0.0, (time.perf_counter() - t) * 1e3
    return _comp_diff(_gray(prev), _gray(cur), ORB_N, RANSAC_PX, t, meta)


def orb_lite_diff(prev, cur, meta=None):
    t = time.perf_counter()
    if prev is None:
        return 0.0, (time.perf_counter() - t) * 1e3
    return _comp_diff(_small(_gray(prev)), _small(_gray(cur)), ORB_LITE_N, RANSAC_PX_LITE, t, meta)


def ov_model_path(model_dir, alias_root=None):
    """Ultralytics chỉ nhận thư mục OpenVINO có '_openvino_model' trong tên; export Kaggle đặt '<name>_openvino_<imgsz>'.
    Không đổi tên thư mục gốc (sha256/TRAIN_LOG theo tên đó) → tạo directory junction '<name>_openvino_model' trỏ tới IR gốc
    trong alias_root (mặc định paths.DERIVED_DIR/ov_alias). Tên đã hợp lệ → trả nguyên."""
    model_dir = Path(model_dir)
    if "_openvino_model" in model_dir.name:
        return model_dir
    if alias_root is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from paths import DERIVED_DIR
        alias_root = DERIVED_DIR / "ov_alias"
    alias_root = Path(alias_root)
    alias_root.mkdir(parents=True, exist_ok=True)
    link = alias_root / f"{model_dir.name}_openvino_model"
    if link.exists() and link.resolve() != model_dir.resolve():
        raise FileExistsError(f"{link} trỏ tới {link.resolve()}, không phải {model_dir}")
    if not link.exists():
        import _winapi
        _winapi.CreateJunction(str(model_dir.resolve()), str(link))
    return link


class TinyDet:
    """Detector nhỏ qua Ultralytics + OpenVINO IR (vd yolo26n@320). score = max conf."""

    def __init__(self, model_dir, imgsz=320, device="CPU"):
        from ultralytics import YOLO
        self.model = YOLO(str(ov_model_path(model_dir)), task="detect")
        self.imgsz, self.device = imgsz, f"intel:{device.lower()}"

    def __call__(self, prev, cur, meta=None):
        t = time.perf_counter()
        r = self.model.predict(cur, imgsz=self.imgsz, device=self.device, conf=0.001, verbose=False)[0]
        s = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        return s, (time.perf_counter() - t) * 1e3


def get_cue(name, **kw):
    if name == "raw_diff":
        return raw_diff
    if name == "ego_comp_diff":
        return ego_comp_diff
    if name == "orb_lite_diff":
        return orb_lite_diff
    if name == "border_band":
        return border_band
    if name == "tiny_det":
        return TinyDet(kw["model_dir"], kw.get("imgsz", 320), kw.get("device", "CPU"))
    raise KeyError(name)


def synth_texture(h=540, w=1024, seed=0):
    """Ảnh tổng hợp có cấu trúc (nhiễu làm mượt + hình chữ nhật) cho test."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w)).astype(np.uint8), (0, 0), 3)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(60):
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 30)
        cv2.rectangle(base, (int(x), int(y)), (int(x + rng.integers(10, 40)), int(y + rng.integers(8, 30))), int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
