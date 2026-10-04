"""Đường dẫn dữ liệu đề tài 28 — MỘT nơi duy nhất (luật Dataset 28-09-2026, 02. Working/CLAUDE.md).

Dataset thô, frames, weights, dump lớn nằm ngoài vùng Google Drive: THS_DATASETS (mặc định $THS_DATASETS).
  UAVDT_DIR    annotations/UAV-benchmark-MOTD_v1.0, annotations/UAVDT_attr, frames/UAV-benchmark-M, _zips/
  VISDRONE_DIR train/, val/, test-dev/ (mỗi split: sequences/, annotations/), _zips/
  MODELS_DIR   yolo26s_visdrone_1024/ (export/ = Kaggle v2, chính thức), yolo26s_visdrone_1024_v1_nms/ (v1, có NMS — không dùng),
               yolo26n_visdrone_1024/ (export/ = Kaggle v3; export_local/…_openvino_320 = IR cue tiny, xuất cục bộ P2E),
               p2_bench/ (IR COCO tạm, chỉ benchmark không có --real), p2_test/ (COCO, chỉ unit test)
  DERIVED_DIR  dumps/<detector>_<imgsz>/<dataset>_<seq>.parquet, cues/, frames_cache/, hits/, audit_short/
Kết quả nhỏ (< 200 MB) vẫn ở ROOT/results. Chạy `python code/paths.py` để xem bảng đường dẫn.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(os.environ.get("THS_DATASETS", str(ROOT.parent / "Datasets")))  # release: set THS_DATASETS

UAVDT_DIR = DATASETS / "UAVDT"
UAVDT_GT = UAVDT_DIR / "annotations" / "UAV-benchmark-MOTD_v1.0"
UAVDT_ATTR = UAVDT_DIR / "annotations" / "UAVDT_attr"
UAVDT_FRAMES = UAVDT_DIR / "frames" / "UAV-benchmark-M"
UAVDT_ZIPS = UAVDT_DIR / "_zips"

VISDRONE_DIR = DATASETS / "VisDrone2019-VID"
VISDRONE_ZIPS = VISDRONE_DIR / "_zips"

MODELS_DIR = DATASETS / "_models" / "28"
BENCH_DIR = MODELS_DIR / "p2_bench"
TEST_MODELS_DIR = MODELS_DIR / "p2_test"
TINY_IR = MODELS_DIR / "yolo26n_visdrone_1024" / "export_local" / "yolo26n_visdrone_1024_openvino_320"  # cue tiny_det

DERIVED_DIR = DATASETS / "_derived" / "28"
DUMPS_DIR = DERIVED_DIR / "dumps"
CUES_DIR = DERIVED_DIR / "cues"
FRAMES_CACHE = DERIVED_DIR / "frames_cache"
HITS_DIR = DERIVED_DIR / "hits"
AUDIT_SHORT_DIR = DERIVED_DIR / "audit_short"

FIXTURES = ROOT / "tests" / "fixtures"


def visdrone_split(split):
    """Thư mục một split VisDrone-VID (chứa sequences/, annotations/).
    Chuẩn: VISDRONE_DIR/<split>; chấp nhận cả tên gốc trong zip VISDRONE_DIR/VisDrone2019-VID-<split>."""
    d = VISDRONE_DIR / split
    alt = VISDRONE_DIR / f"VisDrone2019-VID-{split}"
    return alt if not d.exists() and alt.exists() else d


def model_dir(name):
    """Thư mục weights/export một mô hình (vd. yolo26s_visdrone_1024)."""
    return MODELS_DIR / name


def ensure_layout():
    """Tạo các thư mục gốc còn thiếu (không tạo thư mục dữ liệu con)."""
    for d in (UAVDT_DIR / "annotations", UAVDT_DIR / "frames", UAVDT_ZIPS, VISDRONE_ZIPS, MODELS_DIR,
              DUMPS_DIR, CUES_DIR, FRAMES_CACHE, HITS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe():
    rows = [("THS_DATASETS", DATASETS), ("UAVDT_GT", UAVDT_GT), ("UAVDT_ATTR", UAVDT_ATTR), ("UAVDT_FRAMES", UAVDT_FRAMES),
            ("UAVDT_ZIPS", UAVDT_ZIPS), ("VISDRONE_DIR", VISDRONE_DIR)]
    rows += [(f"VISDRONE[{s}]", visdrone_split(s)) for s in ("train", "val", "test-dev")]
    rows += [("VISDRONE_ZIPS", VISDRONE_ZIPS), ("MODELS_DIR", MODELS_DIR), ("BENCH_DIR", BENCH_DIR), ("TEST_MODELS_DIR", TEST_MODELS_DIR),
             ("TINY_IR", TINY_IR), ("DUMPS_DIR", DUMPS_DIR), ("CUES_DIR", CUES_DIR), ("FRAMES_CACHE", FRAMES_CACHE), ("HITS_DIR", HITS_DIR),
             ("AUDIT_SHORT_DIR", AUDIT_SHORT_DIR), ("FIXTURES", FIXTURES)]
    w = max(len(k) for k, _ in rows)
    for k, p in rows:
        print(f"{k:<{w}}  {'OK ' if p.exists() else '-- '} {p}")
    return rows


if __name__ == "__main__":
    import sys
    if "--ensure" in sys.argv:
        ensure_layout()
    describe()
