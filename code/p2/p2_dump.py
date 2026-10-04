"""P2A-B4: dump detector (OpenVINO IR qua Ultralytics) trên MỌI khung mọi chuỗi → định dạng A1.

paths.DUMPS_DIR/<name>/<dataset>_<seq>.parquet (frame, x, y, w, h, score, cls) + meta.json (t_ms median, sha256 weights).
Tổ hợp dự kiến: yolo26{s,n} × {640, 960, 1280} (paths.MODELS_DIR/<model>_visdrone_1024/export/...). Resume theo chuỗi (bỏ chuỗi đã có file).
Ngưỡng lưu score ≥ --conf-min (mặc định 0.05) để còn chọn ngưỡng sau; không NMS thêm (YOLO26 end-to-end).
Trong lúc chạy in ước lượng thời gian còn lại (median t_ms × khung còn lại); trước khi chạy in ước lượng từ bench_openvino.json nếu có.
Chạy: python code/p2/p2_dump.py --model-dir <IR dir> --name yolo26s_640 --imgsz 640 [--device CPU|GPU] [--dataset uavdt] [--split train]
   hoặc: python code/p2/p2_dump.py --detector yolo26s --imgsz 1024 --split train --device GPU --max-workers 1 --min-free-ram-gb 6
   (--detector → paths.MODELS_DIR/<det>_visdrone_1024/export/<det>_visdrone_1024_openvino_<imgsz>, --name mặc định <det>_<imgsz>).
Luật máy (CLAUDE.md): một worker; trước mỗi chuỗi kiểm RAM trống ≥ --min-free-ram-gb, thiếu → dừng (exit 4), chạy lại để resume.
Split test/all bị khoá tới khi PREREG đóng băng (--allow-test để mở).
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "replay"))
from p2_dump_schema import DUMPS, write_dump, write_meta  # noqa: E402
from p2_frames import frame_paths, seqs_by_split  # noqa: E402
from p2_cues import ov_model_path  # noqa: E402


def sha256_dir(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and p.suffix in (".bin", ".xml", ".onnx", ".pt"):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class OVDetector:
    def __init__(self, model_dir, imgsz, device="CPU", conf_min=0.05):
        from ultralytics import YOLO
        self.model = YOLO(str(ov_model_path(model_dir)), task="detect")  # alias junction nếu tên Kaggle *_openvino_<s>
        self.imgsz, self.device, self.conf = imgsz, f"intel:{device.lower()}", conf_min

    def __call__(self, img):
        t = time.perf_counter()
        r = self.model.predict(img, imgsz=self.imgsz, device=self.device, conf=self.conf, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1e3
        b = r.boxes
        if len(b) == 0:
            return pd.DataFrame(dict(x=[], y=[], w=[], h=[], score=[], cls=np.array([], int))), ms
        xyxy = b.xyxy.cpu().numpy()
        return pd.DataFrame(dict(x=xyxy[:, 0], y=xyxy[:, 1], w=xyxy[:, 2] - xyxy[:, 0], h=xyxy[:, 3] - xyxy[:, 1],
                                 score=b.conf.cpu().numpy(), cls=b.cls.cpu().numpy().astype(int))), ms


def dump_images(det, paths):
    """Chạy detector trên danh sách ảnh (khung 1..n) → (DataFrame dump, list t_ms)."""
    parts, times = [], []
    for f, p in enumerate(paths, start=1):
        d, ms = det(str(p))
        d.insert(0, "frame", f)
        parts.append(d)
        times.append(ms)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["frame", "x", "y", "w", "h", "score", "cls"])
    return df, times


def drop_degenerate(df):
    """Bỏ box suy biến (w ≤ 0 hoặc h ≤ 0 — vd. yolo26n@1024 ở M1202, box bị kẹp sát mép) → (df sạch, số box bỏ)."""
    bad = (df.w <= 0) | (df.h <= 0)
    return df[~bad].reset_index(drop=True), int(bad.sum())


def bench_estimate(name, device):
    p = ROOT / "results" / "p2" / "bench_openvino.json"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    return f"(xem {p.name} cho ms/khung của {name} trên {device})" if txt else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="thư mục IR; bỏ trống nếu dùng --detector")
    ap.add_argument("--detector", default=None, help="vd. yolo26s → export Kaggle trong paths.MODELS_DIR")
    ap.add_argument("--name", default=None, help="mặc định <detector>_<imgsz>")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--dataset", default="uavdt")
    ap.add_argument("--split", default="train", choices=["train", "test", "val", "test-dev", "all"])
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--max-workers", type=int, default=1, help="chỉ 1 (một detector worker trên máy)")
    ap.add_argument("--min-free-ram-gb", type=float, default=6.0)
    ap.add_argument("--allow-test", action="store_true", help="mở split test/all (chỉ sau khi PREREG đóng băng)")
    a = ap.parse_args()
    if a.max_workers != 1:
        sys.exit("--max-workers phải là 1 (không chạy 2 detector worker cùng lúc)")
    if a.split in ("test", "all") and not a.allow_test:
        sys.exit(f"split={a.split} bị khoá tới khi PREREG đóng băng (--allow-test để mở)")
    if a.model_dir is None:
        if a.detector is None:
            sys.exit("cần --model-dir hoặc --detector")
        sys.path.insert(0, str(ROOT / "code"))
        from paths import model_dir
        a.model_dir = model_dir(f"{a.detector}_visdrone_1024") / "export" / f"{a.detector}_visdrone_1024_openvino_{a.imgsz}"
    if a.name is None:
        a.name = f"{a.detector or Path(a.model_dir).name}_{a.imgsz}"
    if not Path(a.model_dir).exists():
        sys.exit(f"không thấy IR: {a.model_dir}")
    det = OVDetector(a.model_dir, a.imgsz, a.device, a.conf_min)
    outdir = DUMPS / a.name
    seqs = {s: sp for s, sp in seqs_by_split(a.dataset).items() if a.split == "all" or sp == a.split}
    todo = [(s, sp) for s, sp in sorted(seqs.items()) if not (outdir / f"{a.dataset}_{s}.parquet").exists()]
    n_left = sum(len(frame_paths(a.dataset, s, sp)) for s, sp in todo)
    print(f"{len(todo)} chuỗi, {n_left} khung còn lại", bench_estimate(a.name, a.device) or "")
    meta_p = outdir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else dict(
        detector=Path(a.model_dir).name, imgsz=a.imgsz, weights_sha256=sha256_dir(a.model_dir), simulated=False, t_ms_median=None,
        n_frames={}, created=dt.datetime.now().isoformat(timespec="seconds"),
        params=dict(device=a.device, conf_min=a.conf_min, model_dir=str(a.model_dir), dataset=a.dataset), t_ms_all_median_by_seq={})
    all_ms = []
    import psutil
    for i, (s, sp) in enumerate(todo):
        free_gb = psutil.virtual_memory().available / 2**30
        if free_gb < a.min_free_ram_gb:
            print(f"DỪNG: RAM trống {free_gb:.1f} GB < {a.min_free_ram_gb} GB trước chuỗi {s}; chạy lại để resume", flush=True)
            sys.exit(4)
        paths = frame_paths(a.dataset, s, sp)
        if not paths:
            print("thiếu frames", s)
            continue
        df, ms = dump_images(det, paths)
        df, n_bad = drop_degenerate(df)
        meta.setdefault("n_degenerate_dropped", {})[s] = n_bad
        if n_bad:
            print(f"  {s}: bỏ {n_bad} box suy biến (w/h ≤ 0)", flush=True)
        write_dump(outdir, a.dataset, s, df, len(paths))
        meta["n_frames"][s] = len(paths)
        meta.setdefault("t_ms_all_median_by_seq", {})[s] = float(np.median(ms))
        all_ms += ms
        meta["t_ms_median"] = float(np.median(all_ms))
        write_meta(outdir, meta)
        n_left -= len(paths)
        print(f"[{i + 1}/{len(todo)}] {s}: {len(df)} box, {np.median(ms):.1f} ms/khung, còn ≈ {n_left * np.median(all_ms) / 3.6e6:.2f} h", flush=True)


if __name__ == "__main__":
    main()
