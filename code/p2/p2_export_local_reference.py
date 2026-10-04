"""P2E-D: xuất cục bộ bản THAM CHIẾU (export_local_reference/) từ weights/best.pt để kiểm chéo export Kaggle (như yolo26s P2C/P2D).

Lệnh như cell export notebook: format openvino (half=True) + onnx (opset 12), nms=False, max_det=400, imgsz ∈ {640, 960, 1024, 1280}.
best.pt được SAO CHÉP sang thư mục tạm tên <name>_<s>.pt rồi mới export (không ghi đè weights/best_openvino_model của Kaggle).
Ghi <model>/export_local_reference/ + EXPORT_LOCAL_REFERENCE.md (lệnh, phiên bản, sha256 best.pt và từng file). Chỉ tham khảo, không dùng cho dump.
Chạy: python code/p2/p2_export_local_reference.py --name yolo26n_visdrone_1024 [--sizes 640 960 1024 1280] --tmp <scratch dir>
"""
import argparse
import hashlib
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paths import model_dir  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1024, 1280])
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--max-det", type=int, default=400)
    a = ap.parse_args()
    from ultralytics import YOLO
    d = model_dir(a.name)
    best = d / "weights" / "best.pt"
    out = d / "export_local_reference"
    assert not out.exists(), f"{out} đã có — không ghi đè"
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    for s in a.sizes:
        pt = tmp / f"{a.name}_{s}.pt"
        shutil.copy2(best, pt)
        m = YOLO(str(pt))
        ir = Path(m.export(format="openvino", half=True, imgsz=s, dynamic=False, nms=False, max_det=a.max_det, device="cpu"))
        ox = Path(YOLO(str(pt)).export(format="onnx", opset=12, imgsz=s, dynamic=False, nms=False, max_det=a.max_det, device="cpu"))
        shutil.move(str(ir), out / ir.name)
        shutil.move(str(ox), out / ox.name)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    lines = [f"# EXPORT_LOCAL_REFERENCE — {a.name} (P2E-D)", "",
             "Chỉ tham khảo (kiểm chéo với export Kaggle `export/`), KHÔNG dùng cho dump.",
             f"Lệnh: `YOLO(<bản sao best.pt>).export(format=openvino, half=True | format=onnx, opset=12; imgsz=s, dynamic=False, nms=False, "
             f"max_det={a.max_det}, device=cpu)`, s ∈ {a.sizes}; ultralytics {md.version('ultralytics')}, openvino {md.version('openvino')}.",
             f"best.pt sha256 = {sha(best)}", "", "## SHA256"]
    lines += [f"{sha(p)}  {p.relative_to(d).as_posix()}" for p in files]
    (d / "EXPORT_LOCAL_REFERENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
