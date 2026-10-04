"""P2E-D: recall theo khung r của từng mức detector trên UAVDT TRAIN (L8) → results/p2/detector_recall_train.json.

r = P(box GT xe VISIBLE (định nghĩa chính P0: occlusion ≠ 2 và out_of_view ≠ 2) ở khung f được ghép với ≥ 1 detection)
  ghép: p2_match (IoU ≥ 0,5, một-một, tham lam) trên MỌI box GT xe; detection = mọi box đã dump có score ≥ score_min.
  P2F-Q1 (01-10-2026, anh Đạt): score_min CHÍNH = 0,25 (mặc định Ultralytics predict = điểm vận hành);
  0,05 (= conf_min của dump) và 0,50 = phụ lục mô tả. Bản 0,05-chính cũ: detector_recall_train_score005.json.
  Chỉ khung trong phạm vi GT (M0207: 1–571). CI 95 %: bootstrap theo chuỗi B = 1000, seed = 42 (L4).
Định dạng r_levels cho p2_estimate_channel / p2_prereg_build: {"yolo26s_1024": r, ...} (score_min chính).
Mặc định --dumps: mọi mức trong LEVELS đã dump đủ chuỗi TRAIN (mức thiếu ghi vào "missing").
Chạy: python code/p2/p2_detector_recall.py [--dumps yolo26s_1024 ...] [--score-min 0.25 0.05 0.5]   (giá trị đầu = chính)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "replay"))
sys.path.insert(0, str(ROOT / "code"))
from p2_match import match_seq  # noqa: E402
from paths import DUMPS_DIR  # noqa: E402

P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p2" / "detector_recall_train.json"
B, SEED = 1000, 42
SCORE_MAIN = 0.25
SCORES = [SCORE_MAIN, 0.05, 0.5]
LEVELS = ["yolo26s_1024", "yolo26n_1024", "yolo26s_640", "yolo26n_640"]


def recall_by_seq(gt, dets, split, score_min):
    """gt: {seq: DataFrame frame,tid,x,y,w,h,visible}; dets: {seq: dump DataFrame} → DataFrame seq, n_vis, n_hit."""
    assert split == "train", f"L8: chỉ TRAIN (nhận split={split!r})"
    rows = []
    for s, g in gt.items():
        d = dets[s]
        h = match_seq(g, d[d.score >= score_min])
        vis = g[g.visible][["tid", "frame"]]
        n_hit = len(vis.merge(h.drop_duplicates(), on=["tid", "frame"]))
        rows.append(dict(seq=s, n_vis=len(vis), n_hit=n_hit))
    return pd.DataFrame(rows)


def pooled_ci(df, B_=B, seed=SEED):
    """r gộp = Σhit / Σvis; CI bootstrap theo chuỗi."""
    rng = np.random.default_rng(seed)
    v, h = df.n_vis.values, df.n_hit.values
    bs = []
    for _ in range(B_):
        i = rng.integers(0, len(df), len(df))
        bs.append(h[i].sum() / max(v[i].sum(), 1))
    return float(h.sum() / v.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", default=None)
    ap.add_argument("--score-min", type=float, nargs="+", default=SCORES)
    a = ap.parse_args()
    split = "train"
    meta = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    seqs = sorted(s for s, m in meta.items() if m["split"] == split)
    dumps = a.dumps or [n for n in LEVELS if all((DUMPS_DIR / n / f"uavdt_{s}.parquet").exists() for s in seqs)]
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["seq", "frame", "tid", "x", "y", "w", "h", "visible"])
    gt = {s: g for s, g in rows[rows.seq.isin(seqs)].groupby("seq")}
    gt = {s: g[g.frame <= meta[s]["n_frames"]] for s, g in gt.items()}
    out = dict(split=split, definition=__doc__.split(chr(10))[2].strip(), iou=0.5, B=B, seed=SEED, levels={}, r_levels={},
               r_levels_by_score={}, missing=[n for n in LEVELS if n not in dumps])
    for name in dumps:
        dets = {s: pd.read_parquet(DUMPS_DIR / name / f"uavdt_{s}.parquet") for s in seqs}
        lv = {}
        for sm in a.score_min:
            df = recall_by_seq(gt, dets, split, sm)
            r, ci = pooled_ci(df)
            lv[f"score>={sm}"] = dict(r=r, ci=ci, n_vis=int(df.n_vis.sum()), n_seq=len(df),
                                      r_seq_median=float((df.n_hit / df.n_vis).median()))
            out["r_levels_by_score"].setdefault(f"{sm:g}", {})[name] = r
            print(f"{name} score>={sm}: r = {r:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
        out["levels"][name] = lv
        out["r_levels"][name] = lv[f"score>={a.score_min[0]}"]["r"]
    out["score_min_main"] = a.score_min[0]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT, "missing:", out["missing"])


if __name__ == "__main__":
    main()
