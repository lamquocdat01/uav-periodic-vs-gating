"""P2 laptop benchmark: YOLO26 OpenVINO inference cost + cheap-cue costs.

Measures, on THIS machine (i7-1185G7 + Iris Xe), with synthetic inputs:
  * yolo26s and yolo26n (COCO weights, temporary stand-ins for the VisDrone models) exported to
    OpenVINO IR FP16 at imgsz 640/960/1280 (static square) = 6 dump combos, + yolo26s@1024
    (denominator of c) + yolo26n@320 (tiny-detector cue); devices CPU and GPU:
    - preprocessing (letterbox 1024x540 -> S, BGR->RGB, HWC->CHW, /255) ms
    - pure inference (compiled_model, LATENCY hint, sync) ms  -> median/p10/p90
    - throughput mode (THROUGHPUT hint, AsyncInferQueue) ms/frame
  * estimated dump hours for N_FRAMES (VisDrone-VID + UAVDT) x 6 combos + recommendation
  * cue costs (OpenCV single-thread and default threads):
    (i) frame diff, (ii) ORB+homography compensated diff, (iii) border-strip diff,
    (iv) yolo26n@320 OpenVINO (CPU/GPU)
  * ratios c = t_cue / t_detector(yolo26s@1024) and / t_detector(yolo26n@320)
  YOLO26 is end-to-end NMS-free, so detector cost does not depend on the number of boxes.

Outputs: results/p2/bench_openvino.json, results/p2/BENCH_OPENVINO.md, results/p2/env_p2.json
Exported models / caches: paths.BENCH_DIR (THS_DATASETS/_models/28/p2_bench, outside Google Drive).

Usage:
  .venv/Scripts/python.exe code/p2_bench_openvino.py [--runs 100] [--warmup 10]
  .venv/Scripts/python.exe code/p2_bench_openvino.py --env-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from paths import BENCH_DIR  # noqa: E402
N_VID = 33_366       # VisDrone-VID frames (P0 inventory)
N_UAVDT = 40_421     # UAVDT frames (P0 inventory)
N_FRAMES = N_VID + N_UAVDT  # 73,787
SRC_W, SRC_H = 1024, 540

PKGS = ["ultralytics", "openvino", "torch", "torchvision", "onnx", "onnxslim", "onnxruntime",
        "opencv-python", "numpy", "pandas", "scipy", "pyarrow", "nbconvert", "nbclient",
        "ipykernel", "nbformat", "psutil"]


# ----------------------------------------------------------------------------- env
def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor()


def ov_devices() -> dict:
    import openvino as ov
    core = ov.Core()
    devs = {}
    for d in core.available_devices:
        info = {}
        for prop in ["FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "OPTIMIZATION_CAPABILITIES",
                     "GPU_DEVICE_TOTAL_MEM_SIZE"]:
            try:
                v = core.get_property(d, prop)
                info[prop] = v if isinstance(v, (str, int, float)) else str(v)
            except Exception:
                pass
        devs[d] = info
    return devs


def env_info() -> dict:
    import cv2
    vers = {}
    for p in PKGS:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    env = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_name": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "packages": vers,
        "cv2_build_threads_default": cv2.getNumThreads(),
    }
    try:
        import torch
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_num_threads"] = torch.get_num_threads()
    except Exception as e:  # noqa
        env["torch_error"] = repr(e)
    try:
        import openvino as ov
        env["openvino_runtime_version"] = ov.get_version()
        env["openvino_available_devices"] = ov.Core().available_devices
        env["openvino_devices"] = ov_devices()
    except Exception as e:  # noqa
        env["openvino_error"] = repr(e)
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        pass
    return env


# ----------------------------------------------------------------------------- utils
def stats(ts_ms: list[float]) -> dict:
    a = np.asarray(ts_ms, dtype=float)
    return {"n": int(a.size), "median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean()), "min_ms": float(a.min())}


def timeit(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def letterbox(img: np.ndarray, s: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    top = (s - nh) // 2
    left = (s - nw) // 2
    return cv2.copyMakeBorder(im, top, s - nh - top, left, s - nw - left, cv2.BORDER_CONSTANT,
                              value=(114, 114, 114))


def preprocess(img: np.ndarray, s: int) -> np.ndarray:
    lb = letterbox(img, s)
    x = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) * (1.0 / 255.0)
    return x[None]


def synth_pair(seed: int = 0):
    """Structured textured 1024x540 frame + second frame shifted 3 px with some moved rectangles."""
    import cv2
    rng = np.random.default_rng(seed)
    pad = 16
    H, W = SRC_H + 2 * pad, SRC_W + 2 * pad
    noise = rng.normal(0, 1, (H, W, 3)).astype(np.float32)
    base = cv2.GaussianBlur(noise, (0, 0), 3.0)
    base += 0.35 * cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32), (0, 0), 0.8)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    static = base.copy()
    rects = []
    for _ in range(40):  # static structures (buildings/roads)
        x, y = int(rng.integers(0, W - 80)), int(rng.integers(0, H - 60))
        w, h = int(rng.integers(20, 80)), int(rng.integers(15, 60))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.rectangle(static, (x, y), (x + w, y + h), c, -1)
    movers = []
    for _ in range(8):  # small moving "vehicles"
        x, y = int(rng.integers(40, W - 60)), int(rng.integers(40, H - 40))
        w, h = int(rng.integers(10, 24)), int(rng.integers(6, 14))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        movers.append((x, y, w, h, c))
    f1 = static.copy()
    for x, y, w, h, c in movers:
        cv2.rectangle(f1, (x, y), (x + w, y + h), c, -1)
    # frame 2: whole background shifted by (3, 0) px (camera ego-motion), movers moved by (6, 4) px extra
    f2 = static.copy()
    for x, y, w, h, c in movers:
        cv2.rectangle(f2, (x + 6, y + 4), (x + 6 + w, y + 4 + h), c, -1)
    dx = 3
    a = f1[pad:pad + SRC_H, pad:pad + SRC_W].copy()
    b = f2[pad:pad + SRC_H, pad - dx:pad - dx + SRC_W].copy()  # content moves +3 px in x
    return a, b


# ----------------------------------------------------------------------------- cues
def make_cues():
    import cv2
    orb = cv2.ORB_create(nfeatures=2000)
    orb500 = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def frame_diff(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(ga, gb)
        _, m = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(m)

    def orb_homog(a, b, info=None, orb=orb, small=False, ransac_px=3.0):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if small:  # ORB-lite (p2_cues.orb_lite_diff): rộng 320 px, 500 điểm, RANSAC 1 px
            ga = cv2.resize(ga, (320, round(ga.shape[0] * 320 / ga.shape[1])), interpolation=cv2.INTER_AREA)
            gb = cv2.resize(gb, (320, round(gb.shape[0] * 320 / gb.shape[1])), interpolation=cv2.INTER_AREA)
        ka, da = orb.detectAndCompute(ga, None)
        kb, db = orb.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 4 or len(kb) < 4:
            return -1
        knn = bf.knnMatch(da, db, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 4:
            return -1
        pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, inl = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)
        if Hm is None:
            return -1
        wa = cv2.warpPerspective(ga, Hm, (gb.shape[1], gb.shape[0]))
        d = cv2.absdiff(wa, gb)
        _, m = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
        if info is not None:
            info.update({"n_kp_a": len(ka), "n_kp_b": len(kb), "n_good": len(good),
                         "n_inliers": int(inl.sum()), "H_tx": float(Hm[0, 2]), "H_ty": float(Hm[1, 2])})
        return cv2.countNonZero(m)

    def border(a, b):
        h, w = a.shape[:2]
        bh, bw = max(1, int(round(0.05 * h))), max(1, int(round(0.05 * w)))
        tot = 0
        for sl in [(slice(0, bh), slice(None)), (slice(h - bh, h), slice(None)),
                   (slice(bh, h - bh), slice(0, bw)), (slice(bh, h - bh), slice(w - bw, w))]:
            ga = cv2.cvtColor(np.ascontiguousarray(a[sl]), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(np.ascontiguousarray(b[sl]), cv2.COLOR_BGR2GRAY)
            d = cv2.absdiff(ga, gb)
            _, m = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
            tot += cv2.countNonZero(m)
        return tot

    def orb_lite(a, b, info=None):
        return orb_homog(a, b, info, orb=orb500, small=True, ransac_px=1.0)

    return {"frame_diff": frame_diff, "orb_homography_diff": orb_homog, "orb_lite_diff": orb_lite,
            "border_strip_diff": border}


# ----------------------------------------------------------------------------- models
def export_ir(weights: str, s: int, mdir: Path) -> Path:
    """Export <weights> to OpenVINO FP16 IR at static imgsz s; cached under mdir."""
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
    stem = Path(weights).stem
    out = mdir / f"{stem}_{s}_e2e_openvino_model"
    if (out / f"{stem}.xml").exists():
        return out
    pt = mdir / weights
    if not pt.exists():
        attempt_download_asset(str(pt))
    # ultralytics 8.4.163: `half` is deprecated -> quantize=16 (FP16 IR); nms=False selects the YOLO26 NMS-free
    # one-to-one head (default nms=None exports the one-to-many head (1, 84, N) that needs external NMS).
    p = YOLO(str(pt)).export(format="openvino", quantize=16, nms=False, imgsz=s, dynamic=False, device="cpu")
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(p), str(out))
    return out


def real_ir(weights: str, s: int, real: dict):
    """IR xuất từ weights VisDrone thật (export/ của paths.model_dir, Kaggle v2) cho model/size, nếu có.
    Nhận cả tên Kaggle `*_openvino_<s>` lẫn tên export cục bộ `*_<s>_openvino_model` (export_local_reference/)."""
    d = real.get(Path(weights).stem)
    if not d:
        return None
    hits = [h for dd in (Path(d), Path(d).parent / "export_local")  # export_local/: IR xuất cục bộ (vd. yolo26n@320 cue tiny)
            for h in sorted(dd.glob(f"*_openvino_{s}")) or sorted(dd.glob(f"*_{s}_openvino_model"))]
    return hits[0] if hits else None


def bench_ir(ir_dir: Path, device: str, img: np.ndarray, s: int, runs: int, warmup: int,
             cache_dir: Path) -> dict:
    import openvino as ov
    xml = next(ir_dir.glob("*.xml"))
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})
    res = {"ir": str(xml), "device": device, "imgsz": s}  # IR ở THS_DATASETS/_models, ngoài ROOT
    model = core.read_model(str(xml))
    t0 = time.perf_counter()
    cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    res["compile_s_latency"] = time.perf_counter() - t0
    for prop in ["INFERENCE_PRECISION_HINT", "INFERENCE_NUM_THREADS", "NUM_STREAMS",
                 "OPTIMAL_NUMBER_OF_INFER_REQUESTS"]:
        try:
            res[prop] = str(cm.get_property(prop))
        except Exception:
            pass
    x = preprocess(img, s)
    req = cm.create_infer_request()
    res["infer_latency"] = timeit(lambda: req.infer({0: x}), runs, warmup)
    res["preprocess"] = timeit(lambda: preprocess(img, s), runs, warmup)
    out = req.get_output_tensor(0).data
    res["output_shape"] = list(out.shape)
    # throughput mode (async queue) - closer to an offline dump
    t0 = time.perf_counter()
    cmt = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    res["compile_s_throughput"] = time.perf_counter() - t0
    nreq = int(cmt.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    q = ov.AsyncInferQueue(cmt, nreq)
    for _ in range(max(warmup, nreq)):
        q.start_async({0: x})
    q.wait_all()
    n_tp = max(runs, 4 * nreq)
    t0 = time.perf_counter()
    for _ in range(n_tp):
        q.start_async({0: x})
    q.wait_all()
    el = time.perf_counter() - t0
    res["throughput"] = {"n_requests": nreq, "frames": n_tp, "fps": n_tp / el, "ms_per_frame": el / n_tp * 1e3}
    return res



def load_snapshot() -> dict:
    """System CPU load + other busy python processes (contention record)."""
    try:
        import psutil
        me = os.getpid()
        procs = []
        for pr in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if pr.info["pid"] == me or "python" not in (pr.info["name"] or "").lower():
                    continue
                cmd = " ".join(pr.info["cmdline"] or [])
                if "-m pip" in cmd:
                    continue
                procs.append({"pid": pr.info["pid"], "cmd": cmd[-120:]})
            except Exception:
                pass
        return {"cpu_percent_2s": psutil.cpu_percent(interval=2.0), "other_python_processes": procs}
    except Exception as e:
        return {"error": repr(e)}

# ----------------------------------------------------------------------------- main
DUMP_MODELS = ["yolo26s.pt", "yolo26n.pt"]   # main detector + low-recall detector
DENOM = ("yolo26s", 1024)                     # c denominator (training resolution)
TINY = ("yolo26n", 320)                       # tiny-detector cue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--models-dir", default=str(BENCH_DIR))
    ap.add_argument("--out", default=str(ROOT / "results" / "p2"))
    ap.add_argument("--env-only", action="store_true")
    ap.add_argument("--real", nargs="*", default=[], metavar="MODEL=DIR",
                    help="dùng IR VisDrone thật, vd. yolo26s=<MODELS_DIR>/yolo26s_visdrone_1024/export (thiếu size → COCO)")
    args = ap.parse_args()
    real = dict(kv.split("=", 1) for kv in args.real)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never let ultralytics pip-install into the venv

    import cv2
    cv2_default_threads = cv2.getNumThreads()  # read BEFORE ultralytics import (it calls cv2.setNumThreads(0))
    env = env_info()
    (out / "env_p2.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"[env] wrote {out / 'env_p2.json'}; OV devices: {env.get('openvino_available_devices')}")
    if args.env_only:
        return

    import cv2
    import openvino as ov
    snap_before = load_snapshot()
    load_before = snap_before.get("cpu_percent_2s")
    t_start = time.perf_counter()
    core = ov.Core()
    avail = core.available_devices
    devices = ["CPU"]
    gpu_note = None
    if any(a.startswith("GPU") for a in avail):
        try:  # probe
            xml = next((real_ir(TINY[0] + ".pt", TINY[1], real) or export_ir("yolo26n.pt", 320, mdir)).glob("*.xml"))
            core.compile_model(core.read_model(str(xml)), "GPU")
            devices.append("GPU")
        except Exception as e:
            gpu_note = f"GPU compile failed: {e!r}"
    else:
        gpu_note = f"GPU not in ov.Core().available_devices={avail}"
    cache_dir = mdir / "ov_cache"
    cache_dir.mkdir(exist_ok=True)

    a, b = synth_pair(0)
    cv2.imwrite(str(mdir / "synth_a.jpg"), a)
    cv2.imwrite(str(mdir / "synth_b.jpg"), b)

    results = {"env": env, "config": {"runs": args.runs, "warmup": args.warmup, "dump_sizes": args.sizes,
                                       "dump_models": [Path(m).stem for m in DUMP_MODELS],
                                       "c_denominator": f"{DENOM[0]}@{DENOM[1]}", "tiny_detector": f"{TINY[0]}@{TINY[1]}",
                                       "src_wh": [SRC_W, SRC_H], "n_frames": N_FRAMES,
                                       "n_frames_breakdown": {"visdrone_vid": N_VID, "uavdt": N_UAVDT},
                                       "cpu_load_percent_before": load_before,
                                       "devices_benchmarked": devices, "gpu_note": gpu_note,
                                       "weights": ("COCO-pretrained yolo26s/yolo26n (temporary; VisDrone weights pending)" if not real
                                                   else {"real_visdrone_ir": real, "others": "COCO-pretrained stand-in"}),
                                       "input": "synthetic 1024x540 textured frame (see synth_pair)"},
               "detector": [], "cues": {}, "ratios": [], "dump_estimate": []}

    # --- detectors: 2 models x dump sizes, + yolo26s@1024 (c denominator) + yolo26n@320 (tiny cue)
    det_cfgs = [(m, s) for m in DUMP_MODELS for s in args.sizes] + [(DENOM[0] + ".pt", DENOM[1]), (TINY[0] + ".pt", TINY[1])]
    for w, s in det_cfgs:
        ir = real_ir(w, s, real) or export_ir(w, s, mdir)
        for d in devices:
            print(f"[det] {w} imgsz={s} device={d} ...", flush=True)
            try:
                r = bench_ir(ir, d, a, s, args.runs, args.warmup, cache_dir)
                r["model"] = Path(w).stem
                r["weights_source"] = "visdrone" if real_ir(w, s, real) else "coco"
                r["total_latency_median_ms"] = r["preprocess"]["median_ms"] + r["infer_latency"]["median_ms"]
                print(f"      pre={r['preprocess']['median_ms']:.2f} ms  infer={r['infer_latency']['median_ms']:.2f} ms"
                      f"  tput={r['throughput']['ms_per_frame']:.2f} ms/f  out={r['output_shape']}", flush=True)
            except Exception as e:
                r = {"model": Path(w).stem, "imgsz": s, "device": d, "error": repr(e)}
                print("      ERROR", e)
            results["detector"].append(r)
    shapes = sorted({str(r["output_shape"]) for r in results["detector"] if "output_shape" in r})
    results["nms_free_note"] = (
        "YOLO26 is end-to-end (one-to-one head, no NMS): exported IR output shapes = " + ", ".join(shapes)
        + " -> fixed top-k detections per frame, so per-frame cost is ~constant w.r.t. the number of objects "
          "(no NMS post-processing term). Timings here exclude any post-processing.")

    # --- JPEG decode proxy (real dump reads JPEG frames from disk)
    ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 95])
    results["jpeg_decode_1024x540"] = timeit(lambda: cv2.imdecode(enc, cv2.IMREAD_COLOR), args.runs, args.warmup)

    # --- cues
    default_threads = cv2_default_threads
    cues = make_cues()
    for label, nth in [("threads_1", 1), (f"threads_default_{default_threads}", default_threads)]:
        cv2.setNumThreads(nth)
        results["cues"][label] = {"cv2_num_threads": cv2.getNumThreads()}
        for name, fn in cues.items():
            results["cues"][label][name] = timeit(lambda fn=fn: fn(a, b), args.runs, args.warmup)
            print(f"[cue] {label} {name}: {results['cues'][label][name]['median_ms']:.3f} ms", flush=True)
    cv2.setNumThreads(default_threads)
    info = {}
    results["cue_sanity"] = {"frame_diff_count": cues["frame_diff"](a, b),
                             "orb_homography_diff_count": cues["orb_homography_diff"](a, b, info),
                             "orb_info": info, "border_strip_diff_count": cues["border_strip_diff"](a, b)}

    # --- ratios c = t_cue / t_detector (detector time = preprocess + latency-mode inference, median)
    def det(model, s, dev):
        m = [r for r in results["detector"] if r.get("model") == model and r["imgsz"] == s and r["device"] == dev
             and "error" not in r]
        return m[0] if m else None

    for dev in devices:
        for ref in (DENOM, TINY):
            rd = det(ref[0], ref[1], dev)
            if rd is None:
                continue
            td, ti = rd["total_latency_median_ms"], rd["infer_latency"]["median_ms"]
            for label, cd in results["cues"].items():
                for name in cues:
                    tc = cd[name]["median_ms"]
                    results["ratios"].append({"denominator": f"{ref[0]}@{ref[1]}/{dev}", "cue": name, "cv2_threads": label,
                                              "t_cue_ms": tc, "t_det_ms": td, "c": tc / td, "c_vs_infer_only": tc / ti})
            if ref == DENOM:  # tiny detector as cue, same device and CPU-tiny vs GPU-denominator
                for dev_t in devices:
                    rt = det(TINY[0], TINY[1], dev_t)
                    if rt is None:
                        continue
                    tc = rt["total_latency_median_ms"]
                    results["ratios"].append({"denominator": f"{ref[0]}@{ref[1]}/{dev}", "cue": f"{TINY[0]}@{TINY[1]}/{dev_t}",
                                              "cv2_threads": "n/a", "t_cue_ms": tc, "t_det_ms": td, "c": tc / td,
                                              "c_vs_infer_only": tc / ti})

    # --- dump estimate (2 models x 3 sizes = 6 combos) per device
    jpeg_ms = results["jpeg_decode_1024x540"]["median_ms"]
    for d in devices:
        per = {}
        for m in DUMP_MODELS:
            for s in args.sizes:
                r = det(Path(m).stem, s, d)
                if r is None:
                    continue
                lat = jpeg_ms + r["preprocess"]["median_ms"] + r["infer_latency"]["median_ms"]
                tp = max(r["throughput"]["ms_per_frame"], jpeg_ms + r["preprocess"]["median_ms"])
                per[f"{Path(m).stem}@{s}"] = {"latency_mode_ms_per_frame": lat,
                                              "throughput_mode_ms_per_frame_lower_bound": tp,
                                              "hours_latency_mode": lat * N_FRAMES / 3.6e6,
                                              "hours_throughput_mode": tp * N_FRAMES / 3.6e6}
        results["dump_estimate"].append({
            "device": d, "per_combo": per,
            "total_hours_latency_mode": sum(v["hours_latency_mode"] for v in per.values()),
            "total_hours_throughput_mode": sum(v["hours_throughput_mode"] for v in per.values())})
    # best device per combo
    combos = {}
    for de in results["dump_estimate"]:
        for k, v in de["per_combo"].items():
            if k not in combos or v["hours_throughput_mode"] < combos[k]["hours_throughput_mode"]:
                combos[k] = dict(v, device=de["device"])
    results["dump_best_device_per_combo"] = combos
    results["dump_best_total_hours_throughput"] = sum(v["hours_throughput_mode"] for v in combos.values())
    results["dump_best_total_hours_latency"] = sum(v["hours_latency_mode"] for v in combos.values())

    results["load_before"] = snap_before
    results["load_after"] = load_snapshot()
    results["bench_wall_s"] = time.perf_counter() - t_start
    (out / "bench_openvino.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_md(results, out / "BENCH_OPENVINO.md")
    print(f"[done] {out / 'bench_openvino.json'}  wall={results['bench_wall_s']:.1f}s")


def prec_str(r: dict) -> str:
    return str(r.get("INFERENCE_PRECISION_HINT", "?")).replace("<Type: '", "").replace("'>", "")


def write_md(R: dict, path: Path):
    e = R["env"]
    C = R["config"]
    L = ["# P2 — Laptop OpenVINO benchmark (auto-generated by code/p2_bench_openvino.py)", "",
         f"- Timestamp: {e['timestamp']}",
         f"- CPU: {e['cpu_name']} ({e['cpu_logical']} logical); OV devices: {e.get('openvino_available_devices')}",
         f"- GPU: {e.get('openvino_devices', {}).get('GPU', {}).get('FULL_DEVICE_NAME', 'n/a')}"
         + (f" — NOTE: {C['gpu_note']}" if C['gpu_note'] else ""),
         f"- openvino {e['packages'].get('openvino')}, ultralytics {e['packages'].get('ultralytics')}, "
         f"opencv {e['packages'].get('opencv-python')}, torch {e['packages'].get('torch')}",
         f"- {C['runs']} timed runs after {C['warmup']} warm-ups; CPU load in the 2 s before the bench: "
         f"{C['cpu_load_percent_before']}% (other analysis jobs were running concurrently on this laptop → numbers are "
         "noisy; where throughput-mode ms > latency-mode ms the CPU was contended during that measurement). "
         f"Other python processes at start: {len(R.get('load_before', {}).get('other_python_processes', []))}, "
         f"CPU% after bench: {R.get('load_after', {}).get('cpu_percent_2s')} (full list in json `load_before`/`load_after`).",
         "- **Limitation:** inputs are SYNTHETIC 1024×540 frames (smoothed noise + rectangles). Detector cost is "
         "content-independent, but ORB/homography cost depends on texture/keypoints; re-run on real UAVDT frames when available.",
         f"- Weights: {C['weights']} — VisDrone-trained weights have the same architecture/cost except the 10-class head.",
         f"- {R['nms_free_note']}", "",
         "## Detector (OpenVINO IR FP16, static square input, LATENCY hint, sync infer)", "",
         "| model@imgsz | device | exec precision | preprocess med ms | infer med ms (p10–p90) | pre+infer ms | throughput-mode ms/frame (nreq) |",
         "|---|---|---|---|---|---|---|"]
    for r in R["detector"]:
        if "error" in r:
            L.append(f"| {r['model']}@{r['imgsz']} | {r['device']} | – | ERROR: {r['error']} | | | |")
            continue
        il = r["infer_latency"]
        L.append(f"| {r['model']}@{r['imgsz']} | {r['device']} | {prec_str(r)} | "
                 f"{r['preprocess']['median_ms']:.2f} | {il['median_ms']:.2f} ({il['p10_ms']:.2f}–{il['p90_ms']:.2f}) | "
                 f"{r['total_latency_median_ms']:.2f} | {r['throughput']['ms_per_frame']:.2f} ({r['throughput']['n_requests']}) |")
    L += ["", f"JPEG decode 1024×540 (proxy for reading real frames): {R['jpeg_decode_1024x540']['median_ms']:.2f} ms median.", "",
          f"## Dump estimate — {C['n_frames']:,} frames (VisDrone-VID {N_VID:,} + UAVDT {N_UAVDT:,}) × 6 combos", "",
          "Per-frame latency mode = JPEG decode + preprocess + inference, sequential. Throughput mode = async queue "
          "(lower bound; assumes decode/preprocess overlapped with inference). Output serialisation not included.", "",
          "| device | combo | ms/frame latency | h latency | ms/frame throughput | h throughput |", "|---|---|---|---|---|---|"]
    for de in R["dump_estimate"]:
        for k, v in de["per_combo"].items():
            L.append(f"| {de['device']} | {k} | {v['latency_mode_ms_per_frame']:.1f} | {v['hours_latency_mode']:.2f} | "
                     f"{v['throughput_mode_ms_per_frame_lower_bound']:.1f} | {v['hours_throughput_mode']:.2f} |")
        L.append(f"| **{de['device']} total (6 combos)** | | | **{de['total_hours_latency_mode']:.2f}** | | "
                 f"**{de['total_hours_throughput_mode']:.2f}** |")
    bc = R["dump_best_device_per_combo"]
    if bc:
        hb, hl = R["dump_best_total_hours_throughput"], R["dump_best_total_hours_latency"]
        order = sorted(bc.items(), key=lambda kv: (0 if kv[0].startswith("yolo26s") else 1, kv[1]["hours_throughput_mode"]))
        L += ["", f"Best device per combo: " + ", ".join(f"{k}→{v['device']} ({v['hours_throughput_mode']:.2f} h)" for k, v in bc.items())
              + f". Total ≈ **{hb:.1f} h** (throughput mode) / {hl:.1f} h (latency mode).", ""]
        nights = int(np.ceil(hb / 8.0))
        if hl <= 8:
            fit = f"All 6 combos fit in one ~8 h overnight run even sequentially (≈{hl:.1f} h latency mode, ≈{hb:.1f} h async). "
        elif hb <= 8:
            fit = (f"≈{hb:.1f} h with async/throughput mode (lower bound) but ≈{hl:.1f} h sequential: plan ONE night only if the "
                   f"async pipeline is used, otherwise TWO batches (e.g. night 1 = all yolo26s combos, night 2 = all yolo26n combos). ")
        else:
            fit = f"Does not fit one night (≈{hb:.1f} h async): split into ~{nights} overnight batches of ≤8 h. "
        L += ["**Recommendation.** " + fit
              + "Run order (main detector first, cheapest first so the pipeline is validated early): "
              + " → ".join(f"{k} ({v['device']}, {v['hours_throughput_mode']:.1f} h)" for k, v in order)
              + ". Use async/throughput mode, write one output file per (combo, sequence) so every batch is resumable, "
                "keep the laptop on AC power with sleep disabled, and avoid running the CPU cue pass concurrently "
                "with a CPU detector batch (shared cores)."]
    L += ["", "## Cue costs (synthetic pair: 3 px global shift + 8 moved rectangles), median ms (p90)", "",
          "| cue | " + " | ".join(R["cues"].keys()) + " |", "|---|" + "---|" * len(R["cues"])]
    names = [k for k in next(iter(R["cues"].values())).keys() if k != "cv2_num_threads"]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{v[n]['median_ms']:.3f} ({v[n]['p90_ms']:.3f})" for v in R["cues"].values()) + " |")
    for r in R["detector"]:
        if (r.get("model"), r.get("imgsz")) == TINY and "error" not in r:
            L.append(f"| {TINY[0]}@{TINY[1]} OV {r['device']} (pre+infer) | {r['total_latency_median_ms']:.3f} | (n/a) |")
    s = R["cue_sanity"]
    oi = s["orb_info"]
    L += ["", f"Sanity: frame-diff changed px = {s['frame_diff_count']}, ORB-compensated = {s['orb_homography_diff_count']} "
              f"(H tx={oi.get('H_tx', float('nan')):.2f}, ty={oi.get('H_ty', float('nan')):.2f}, "
              f"inliers {oi.get('n_inliers')}/{oi.get('n_good')}), border strip = {s['border_strip_diff_count']}.",
          "", "## Ratios c = t_cue / t_detector (detector = preprocess + inference median; OpenCV cues single-thread)", "",
          "| denominator | frame_diff | orb_homography_diff | orb_lite_diff | border_strip_diff | yolo26n@320/CPU | yolo26n@320/GPU |",
          "|---|---|---|---|---|---|---|"]
    dens = []
    for r in R["ratios"]:
        if r["denominator"] not in dens:
            dens.append(r["denominator"])
    for dn in dens:
        row = []
        for cue in ["frame_diff", "orb_homography_diff", "orb_lite_diff", "border_strip_diff", "yolo26n@320/CPU", "yolo26n@320/GPU"]:
            m = [r for r in R["ratios"] if r["denominator"] == dn and r["cue"] == cue and r["cv2_threads"] in ("threads_1", "n/a")]
            row.append(f"{m[0]['c']:.4f}" if m else "–")
        L.append(f"| {dn} | " + " | ".join(row) + " |")
    L += ["", "Default-thread cue ratios and infer-only ratios are in bench_openvino.json (`ratios`).", "",
          f"Bench wall time: {R['bench_wall_s']:.0f} s."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
