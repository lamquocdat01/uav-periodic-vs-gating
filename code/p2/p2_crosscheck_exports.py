"""P2D-B3: kiểm chéo export chính thức (Kaggle v2, export/) với bản xuất cục bộ (export_local_reference/).

Cùng weights (best.pt sha256 giống), cùng opset → đầu ra phải gần như trùng. Với mỗi imgsz và mỗi định dạng
(IR OpenVINO FP16, ONNX chạy qua OpenVINO CPU), chạy N ảnh tổng hợp (p2_verify_weights.synth_image, seed 0..N-1):
  - số box conf >= 0.25 và >= 0.05 giữa hai bản phải bằng nhau;
  - top-10 hàng theo score: |Δ toạ độ| (pixel ảnh đầu vào), |Δ score| < tol (mặc định 1e-3); lớp phải trùng.
P2E: --name yolo26n_visdrone_1024 (cần export_local_reference/ của n); --real → N khung UAVDT TRAIN THẬT (seed 42, chia đều
theo chuỗi, p2_bench_c_real.sample_pairs; TEST cấm) thay ảnh tổng hợp.
Usage: .venv/Scripts/python.exe code/p2/p2_crosscheck_exports.py [--name yolo26s_visdrone_1024] [--real] [--n 20] [--tol 1e-3]
       [--out results/p2/crosscheck_exports.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paths import model_dir  # noqa: E402
from p2_verify_weights import letterbox, synth_image  # noqa: E402

NAME = "yolo26s_visdrone_1024"


def real_frames(n):
    """n khung UAVDT TRAIN thật (ndarray BGR) — khung sau của các cặp p2_bench_c_real.sample_pairs."""
    import cv2
    from p2_bench_c_real import sample_pairs
    from p2_frames import frame_paths, seqs_by_split
    sf = {s: frame_paths("uavdt", s) for s, sp in seqs_by_split("uavdt").items() if sp == "train"}
    return [cv2.imread(str(p1)) for _, _, _, p1 in sample_pairs(sf, n, "train")]


def pairs(d: Path, s: int, NAME=NAME):
    off, loc = d / "export", d / "export_local_reference"
    yield "openvino", off / f"{NAME}_openvino_{s}" / "best.xml", loc / f"{NAME}_{s}_openvino_model" / f"{NAME}_{s}.xml"
    yield "onnx", off / f"{NAME}_{s}.onnx", loc / f"{NAME}_{s}.onnx"


def run(cm, img, s):
    x = letterbox(img, s, s)[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
    return cm({0: x})[cm.outputs[0]][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1024, 1280])
    ap.add_argument("--out", default=str(ROOT / "results" / "p2" / "crosscheck_exports.json"))
    a = ap.parse_args()
    import openvino as ov
    core = ov.Core()
    d = model_dir(a.name)
    imgs = real_frames(a.n) if a.real else [synth_image(seed) for seed in range(a.n)]
    rows, ok = [], True
    for s in a.sizes:
        for fmt, f_off, f_loc in pairs(d, s, a.name):
            c_off, c_loc = core.compile_model(str(f_off), "CPU"), core.compile_model(str(f_loc), "CPU")
            r = {"imgsz": s, "format": fmt, "n_img": a.n, "count_mismatch": 0, "cls_mismatch": 0,
                 "max_dxyxy": 0.0, "max_dscore": 0.0, "max_score_seen": 0.0}
            for img in imgs:
                p, q = run(c_off, img, s), run(c_loc, img, s)
                for th in (0.25, 0.05):
                    r["count_mismatch"] += int((p[:, 4] >= th).sum() != (q[:, 4] >= th).sum())
                i, j = np.argsort(-p[:, 4], kind="stable")[:10], np.argsort(-q[:, 4], kind="stable")[:10]
                r["cls_mismatch"] += int((p[i, 5] != q[j, 5]).sum())
                r["max_dxyxy"] = max(r["max_dxyxy"], float(np.abs(p[i, :4] - q[j, :4]).max()))
                r["max_dscore"] = max(r["max_dscore"], float(np.abs(p[i, 4] - q[j, 4]).max()))
                r["max_score_seen"] = max(r["max_score_seen"], float(p[i[0], 4]))
            r["pass"] = r["count_mismatch"] == 0 and r["cls_mismatch"] == 0 and r["max_dxyxy"] < a.tol and r["max_dscore"] < a.tol
            ok &= r["pass"]
            rows.append(r)
            print(f"[{'PASS' if r['pass'] else 'FAIL'}] {fmt:8s} {s:4d}: count_mismatch {r['count_mismatch']}, cls_mismatch "
                  f"{r['cls_mismatch']}, max|dxyxy| {r['max_dxyxy']:.3g}, max|dscore| {r['max_dscore']:.3g}, top score {r['max_score_seen']:.3f}")
    Path(a.out).write_text(json.dumps({"name": a.name, "input": "uavdt_train_real" if a.real else "synthetic", "tol": a.tol,
                                       "all_pass": ok, "rows": rows}, indent=2), encoding="utf-8")
    print("[RESULT]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
