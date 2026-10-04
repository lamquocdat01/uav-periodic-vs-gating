"""P2A-B2: chạy cue trên MỌI khung của mỗi chuỗi → paths.CUES_DIR/<dataset>_<seq>.parquet (frame, cue, score, t_ms[, H_ok]).

Resume theo chuỗi (bỏ qua chuỗi đã có đủ cue yêu cầu). Chạy được trên cả TRAIN và TEST (TEST chỉ để replay);
ước lượng kênh (q_in, q_out, q_b, q_o, c) CHỈ dùng TRAIN — xem p2_estimate_channel.py (L8).
Chạy: python code/p2/p2_cue_trace.py --dataset uavdt [--split train|test|all] [--cues raw_diff,ego_comp_diff,orb_lite_diff,border_band,tiny_det]
      [--tiny-dir paths.TINY_IR (yolo26n VisDrone @320, mặc định)] [--limit N]
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from p2_cues import CUES, get_cue  # noqa: E402
from p2_frames import frame_paths, seqs_by_split  # noqa: E402

sys.path.insert(0, str(ROOT / "code"))
from paths import CUES_DIR as OUT, TINY_IR  # noqa: E402


def trace_seq(dataset, seq, split, cues, fns, limit=None):
    imgs = frame_paths(dataset, seq, split)[:limit] if limit else frame_paths(dataset, seq, split)
    if not imgs:
        return None
    recs, prev = [], None
    for f, p in enumerate(imgs, start=1):
        cur = cv2.imread(str(p))
        for c in cues:
            meta = {}
            s, t = fns[c](prev, cur, meta)
            recs.append(dict(frame=f, cue=c, score=float(s), t_ms=float(t), H_ok=meta.get("H_ok")))
        prev = cur
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uavdt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cues", default="raw_diff,ego_comp_diff,border_band")
    ap.add_argument("--tiny-dir", default=str(TINY_IR))
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cues = [c for c in a.cues.split(",") if c]
    assert all(c in CUES for c in cues), cues
    fns = {c: get_cue(c, model_dir=a.tiny_dir, device=a.device) for c in cues}
    OUT.mkdir(parents=True, exist_ok=True)
    todo = {s: sp for s, sp in seqs_by_split(a.dataset).items() if a.split == "all" or sp == a.split}
    n_done = n_skip = n_missing = 0
    for seq, sp in sorted(todo.items()):
        out = OUT / f"{a.dataset}_{seq}.parquet"
        if out.exists() and set(cues) <= set(pd.read_parquet(out, columns=["cue"]).cue.unique()):
            n_skip += 1
            continue
        df = trace_seq(a.dataset, seq, sp, cues, fns, a.limit)
        if df is None:
            n_missing += 1
            continue
        if out.exists():  # giữ cue cũ, thay cue chạy lại
            old = pd.read_parquet(out)
            df = pd.concat([old[~old.cue.isin(cues)], df], ignore_index=True)
        df.to_parquet(out, index=False)
        n_done += 1
        print(seq, sp, len(df), flush=True)
    print(f"done {n_done}, skipped {n_skip}, no frames {n_missing}")


if __name__ == "__main__":
    main()
