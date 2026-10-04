"""P2A-A2: detector GIẢ LẬP từ GT — "detector lý tưởng có per-look recall r".

Mỗi box GT VISIBLE (định nghĩa chính) được phát hiện độc lập với xác suất r (Bernoulli, A-ind),
box = đúng box GT (IoU = 1), score = 0.9. Thêm false positive: số FP mỗi khung ~ Poisson(fp_rate),
vị trí đều trong ảnh, kích thước lấy ngẫu nhiên từ box GT của chuỗi, score ~ U(0.3, 0.9), cls = -1.
Seed = 42 + crc32(seq) (tái lập theo chuỗi, độc lập giữa các chuỗi).
Xuất paths.DUMPS_DIR/sim_r<r>_fp<fp>/<dataset>_<seq>.parquet + meta.json (simulated=True).
Chạy: python code/p2/replay/p2_sim_detector.py [--split train|test|all]
"""
import argparse
import datetime as dt
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from p2_dump_schema import DUMPS, write_dump, write_meta

ROOT = Path(__file__).resolve().parents[3]
P0 = ROOT / "results" / "p0"
SEED = 42
R_GRID = [1.0, 0.8, 0.5]
FP_GRID = [0.0, 0.05]


def seq_seed(seq, salt=0):
    return SEED + zlib.crc32(f"{seq}|{salt}".encode()) % (2 ** 31)


def dump_name(r, fp):
    return f"sim_r{r:g}_fp{fp:g}"


def simulate_seq(rows_seq, n_frames, W, H, r, fp_rate, seq):
    rng = np.random.default_rng(seq_seed(seq))
    vis = rows_seq[rows_seq.visible]
    keep = rng.random(len(vis)) < r
    det = vis.loc[keep, ["frame", "x", "y", "w", "h"]].copy()
    det["score"] = 0.9
    det["cls"] = 0
    parts = [det]
    if fp_rate > 0:
        rng_fp = np.random.default_rng(seq_seed(seq, "fp"))
        k = rng_fp.poisson(fp_rate, n_frames)
        n = int(k.sum())
        if n:
            frames = np.repeat(np.arange(1, n_frames + 1), k)
            idx = rng_fp.integers(0, len(vis), n)
            w, h = vis.w.values[idx], vis.h.values[idx]
            parts.append(pd.DataFrame(dict(frame=frames, x=rng_fp.uniform(0, W - w), y=rng_fp.uniform(0, H - h), w=w, h=h,
                                           score=rng_fp.uniform(0.3, 0.9, n), cls=-1)))
    return pd.concat(parts, ignore_index=True).sort_values("frame", kind="stable")


def main(split="train"):
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "visible", "group"])
    rows = rows[rows.group == "VEHICLE"]
    if split != "all":
        rows = rows[rows.split == split]
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    nfr = fr.groupby("seq").frame.max().to_dict()
    meta_p = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    for r in R_GRID:
        for fp in FP_GRID:
            d = DUMPS / dump_name(r, fp)
            n = {}
            for seq, g in rows.groupby("seq"):
                df = simulate_seq(g, nfr[seq], meta_p[seq]["W"], meta_p[seq]["H"], r, fp, seq)
                write_dump(d, "uavdt", seq, df, nfr[seq])
                n[seq] = int(nfr[seq])
            write_meta(d, dict(detector="sim_gt", imgsz=None, weights_sha256=None, simulated=True, t_ms_median=None, n_frames=n,
                               created=dt.datetime.now().isoformat(timespec="seconds"),
                               params=dict(r=r, fp_rate=fp, seed=SEED, split=split, source="GT VISIBLE chính (UAVDT)",
                                           note="detector lý tưởng: box = GT, Bernoulli(r) độc lập theo box; FP Poisson/khung")))
            print(d.name, len(n), "seqs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    main(ap.parse_args().split)
