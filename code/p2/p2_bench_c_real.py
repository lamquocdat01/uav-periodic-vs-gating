"""P2E-B1': đo c = t_cue / t_detector trên KHUNG UAVDT TRAIN THẬT (thay cặp ảnh tổng hợp của p2_bench_openvino).

Mẫu: N cặp (khung f−1, f) từ các chuỗi TRAIN (L8; assert split == "train"), chia đều theo chuỗi, f chọn ngẫu nhiên seed=42.
Ảnh giải mã sẵn vào RAM → thời gian KHÔNG gồm đọc jpeg (cả cue lẫn detector nhận ndarray BGR).
Mỗi lần lặp (≥ 3): warmup rồi đo từng cặp cho
  cue p2_cues: raw_diff, ego_comp_diff, orb_lite_diff, border_band (cv2 1 luồng = số chính, và cv2 mặc định),
       tiny_det yolo26n@320 VisDrone (CPU, GPU);
  detector (Ultralytics predict, như p2_dump nhưng từ ndarray): yolo26s@1024, yolo26n@1024 (GPU = mẫu số chính, CPU tham khảo).
t_X(lần) = median theo cặp; c(lần) = t_cue / t_det; báo median theo các lần + [min, max] (độ nhiễu giữa các lần).
Ghi trạng thái máy (CPU %, RAM trống) trước mỗi lần.
Chạy: python code/p2/p2_bench_c_real.py [--n 200] [--repeats 3] [--warmup 10] [--out results/p2/bench_real/bench_c_real.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code"))
from p2_cues import TinyDet, border_band, ego_comp_diff, orb_lite_diff, raw_diff  # noqa: E402
from p2_frames import frame_paths, seqs_by_split  # noqa: E402
from paths import MODELS_DIR, TINY_IR  # noqa: E402

SEED = 42
CV_CUES = {"raw_diff": raw_diff, "ego_comp_diff": ego_comp_diff, "orb_lite_diff": orb_lite_diff, "border_band": border_band}
DETS = [("yolo26s", 1024), ("yolo26n", 1024)]
DENOM = ("yolo26s_1024", "GPU")


def sample_pairs(seqs_frames, n, split, seed=SEED):
    """seqs_frames: {seq: [paths]} CHỈ TRAIN → list (seq, f, path_prev, path_cur), f ≥ 2 (1-based), chia đều theo chuỗi."""
    assert split == "train", f"L8: chỉ TRAIN (nhận split={split!r})"
    rng = np.random.default_rng(seed)
    seqs = sorted(s for s, p in seqs_frames.items() if len(p) >= 2)
    per = [n // len(seqs) + (1 if i < n % len(seqs) else 0) for i in range(len(seqs))]
    out = []
    for s, k in zip(seqs, per):
        ps = seqs_frames[s]
        for f in sorted(rng.choice(np.arange(2, len(ps) + 1), size=min(k, len(ps) - 1), replace=False)):
            out.append((s, int(f), ps[f - 2], ps[f - 1]))
    return out


def time_pairs(fn, pairs, warmup):
    """fn(prev, cur) → trả thời gian ms (tự đo); median theo cặp sau warmup (warmup lặp trên các cặp đầu)."""
    for i in range(warmup):
        fn(*pairs[i % len(pairs)])
    return float(np.median([fn(a, b) for a, b in pairs]))


def summarize(runs):
    """runs: list {key: ms} theo lần lặp → {key: {median, min, max, runs}}."""
    keys = runs[0].keys()
    return {k: dict(median=float(np.median([r[k] for r in runs])), min=float(min(r[k] for r in runs)),
                    max=float(max(r[k] for r in runs)), runs=[r[k] for r in runs]) for k in keys}


def ratios(runs, denom_key):
    """c theo từng lần (cùng lần đo tử và mẫu) → median + [min, max]."""
    out = {}
    for k in runs[0]:
        if k == denom_key:
            continue
        cs = [r[k] / r[denom_key] for r in runs]
        out[k] = dict(c_median=float(np.median(cs)), c_min=float(min(cs)), c_max=float(max(cs)), c_runs=cs)
    return out


def machine_state():
    import psutil
    return dict(cpu_pct_5s=psutil.cpu_percent(interval=5), ram_free_gb=psutil.virtual_memory().available / 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "p2" / "bench_real" / "bench_c_real.json"))
    a = ap.parse_args()
    split = "train"
    sf = {s: frame_paths("uavdt", s) for s, sp in seqs_by_split("uavdt").items() if sp == split}
    pairs_meta = sample_pairs(sf, a.n, split)
    assert len(pairs_meta) >= a.n, f"chỉ có {len(pairs_meta)} cặp"
    pairs = [(cv2.imread(str(p0)), cv2.imread(str(p1))) for _, _, p0, p1 in pairs_meta]
    print(f"{len(pairs)} cặp TRAIN từ {len({m[0] for m in pairs_meta})} chuỗi, ảnh {pairs[0][1].shape}", flush=True)

    tiny = {d: TinyDet(TINY_IR, 320, d) for d in ("CPU", "GPU")}
    dets = {}
    for det, s in DETS:
        for dev in ("GPU", "CPU"):
            from p2_dump import OVDetector
            ir = MODELS_DIR / f"{det}_visdrone_1024" / "export" / f"{det}_visdrone_1024_openvino_{s}"
            dets[(f"{det}_{s}", dev)] = OVDetector(ir, s, dev)

    def timed(f):
        def g(prev, cur):
            t = time.perf_counter()
            f(prev, cur)
            return (time.perf_counter() - t) * 1e3
        return g

    runs, states = [], []
    default_threads = cv2.getNumThreads()
    for r in range(a.repeats):
        states.append(machine_state())
        print(f"run {r + 1}/{a.repeats} state {states[-1]}", flush=True)
        m = {}
        for thr, label in ((1, "1thr"), (default_threads, "default")):
            cv2.setNumThreads(thr)
            for name, fn in CV_CUES.items():
                m[f"{name}/{label}"] = time_pairs(timed(fn), pairs, a.warmup)
        cv2.setNumThreads(default_threads)
        for dev, td in tiny.items():
            m[f"tiny_det_yolo26n_320/{dev}"] = time_pairs(timed(td), pairs, a.warmup)
        for (name, dev), det in dets.items():
            m[f"{name}/{dev}"] = time_pairs(timed(lambda p, c, det=det: det(c)), pairs, a.warmup)
        runs.append(m)
        print(json.dumps({k: round(v, 2) for k, v in m.items()}), flush=True)

    denom = f"{DENOM[0]}/{DENOM[1]}"
    out = dict(split=split, n_pairs=len(pairs), n_seqs=len({m[0] for m in pairs_meta}), seed=SEED, repeats=a.repeats,
               warmup=a.warmup, image_shape=list(pairs[0][1].shape), cv2_default_threads=default_threads,
               timing="perf_counter quanh lời gọi, ảnh đã giải mã (không gồm đọc jpeg); detector = Ultralytics predict (pre+infer+post)",
               denominator=denom, machine_state=states, t_ms=summarize(runs), c=ratios(runs, denom),
               c_vs_yolo26s_1024_CPU=ratios(runs, "yolo26s_1024/CPU"),
               pairs=[dict(seq=s, frame=f) for s, f, _, _ in pairs_meta])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: round(v["c_median"], 4) for k, v in out["c"].items()}, indent=1))


if __name__ == "__main__":
    main()
