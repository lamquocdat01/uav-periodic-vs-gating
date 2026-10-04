"""P3-A1–A3 (sau freeze prereg-28-v1): chuẩn bị UAVDT TEST theo đúng các bước đã đóng băng — KHÔNG ước lượng lại gì.

A1 Luật dữ liệu PREREG §1: kích thước ảnh thật từng chuỗi (parse_uavdt.json, size_src = image); khung ngoài GT cuối bị loại
   (frames_uavdt.parquet chỉ chứa khung trong phạm vi GT) → bảng chuỗi (W × H, số ảnh, số khung dùng, khung GT cuối).
A2 Ego-motion TEST: cùng thủ tục P0 (p0_egomotion.process_seq: ORB + RANSAC, cặp mỗi 10 khung, ≤ 300 cặp, median theo chuỗi);
   nhãn hovering/moving bằng ngưỡng ĐÃ ĐÓNG BĂNG trong results/p0/hover_threshold.json (0,61 px/khung) — không chạy lại GMM.
   Ghi riêng results/p3/egomotion_seq_test.parquet (KHÔNG ghi đè egomotion_seq.parquet của TRAIN, đầu vào đóng băng của builder).
   Mật độ M theo GT (M_at_birth) → bin 6–20 / > 20 (1–5 chỉ mô tả).
A3 E3 TEST theo định nghĩa P0 (events_E3_uavdt.parquet, VISIBLE chính, cửa sổ sinh W = 120 như builder/engine); đếm theo ego × M_bin.
   Audit pre-birth TEST = p2_audit_prebirth.py --split test --allow-test (đóng băng; chạy riêng, cần dump yolo26s_1024 TEST).
Xuất: results/p3/test_prepare.json, results/p3/test_sequences.csv.
Chạy: python code/p3/p3_prepare_test.py [--skip-ego]   (--skip-ego: đọc lại egomotion_seq_test.parquet đã có)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
P0 = ROOT / "results" / "p0"
OUT = ROOT / "results" / "p3"
SPLIT = "test"
W_BIRTH = 120


def ego_test(skip=False):
    p = OUT / "egomotion_seq_test.parquet"
    if skip and p.exists():
        return pd.read_parquet(p)
    import cv2
    import p0_egomotion as eg
    orb, bf = cv2.ORB_create(nfeatures=2000), cv2.BFMatcher(cv2.NORM_HAMMING)
    recs = []
    srcs = [x for x in eg.seq_sources("uavdt", SPLIT) if x[1] == SPLIT]
    for i, (seq, sp, imgs, boxes) in enumerate(srcs):
        recs += eg.process_seq("uavdt", seq, sp, imgs, boxes, orb, bf)
        print(f"  ego [{i + 1}/{len(srcs)}] {seq}", flush=True)
    s = pd.DataFrame(recs)
    s.to_parquet(OUT / "egomotion_samples_test.parquet", index=False)
    seq = s.groupby(["dataset", "split", "seq"]).agg(
        n_pairs=("frame", "size"), H_ok_rate=("H_ok", "mean"), shift_px_median=("shift_px", "median"),
        inlier_ratio_median=("inlier_ratio", "median"), W=("W", "first"), H=("H", "first")).reset_index()
    seq.to_parquet(p, index=False)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ego", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    seqs = sorted(s for s, m in meta.items() if m["split"] == SPLIT)
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    fr = fr[fr.split == SPLIT]
    n_used = fr.groupby("seq").frame.max()
    # A2
    hv = json.loads((P0 / "hover_threshold.json").read_text(encoding="utf-8"))["by_dataset"]["UAVDT"]
    thr = float(hv["proposed_threshold_px"])
    eg = ego_test(a.skip_ego)
    eg["ego"] = np.where(eg.shift_px_median < thr, "hovering", "moving")
    ego = dict(zip(eg.seq, eg.ego))
    # A1
    tab = pd.DataFrame([dict(seq=s, W=meta[s]["W"], H=meta[s]["H"], size_src=meta[s]["size_src"], gt_extent_x=meta[s]["gt_extent"][0],
                             gt_extent_y=meta[s]["gt_extent"][1], n_img=meta[s]["n_img"], gt_max_frame=meta[s]["gt_max_frame"],
                             n_frames_used=int(n_used.get(s, 0)), n_frames_dropped=int(meta[s]["n_img"] - n_used.get(s, 0)),
                             shift_px_median=float(eg.set_index("seq").shift_px_median.get(s, np.nan)), ego=ego.get(s, "unknown"),
                             M_median=float(fr[fr.seq == s].M.median()))
                        for s in seqs])
    # A3
    last = fr.groupby("seq").frame.max()
    e3 = pd.read_parquet(P0 / "events_E3_uavdt.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main") & (e3.split == SPLIT)]
    n_all = len(e3)
    e3w = e3[e3.start <= e3.seq.map(last) - W_BIRTH].copy()
    e3w["M_bin"] = pd.cut(e3w.M_at_birth, [-np.inf, 5.5, 20.5, np.inf], labels=["1-5", "6-20", ">20"]).astype(str)
    e3w["ego"] = e3w.seq.map(ego)
    strata = [dict(ego=e, M_bin=m, n_events=int(len(g)), n_seq=int(g.seq.nunique()), size_rule=bool(len(g) >= 30 and g.seq.nunique() >= 3))
              for (e, m), g in e3w.groupby(["ego", "M_bin"])]
    tab["n_E3_all"] = tab.seq.map(e3.groupby("seq").size()).fillna(0).astype(int)
    tab["n_E3_birth_window"] = tab.seq.map(e3w.groupby("seq").size()).fillna(0).astype(int)
    tab.to_csv(OUT / "test_sequences.csv", index=False)
    out = dict(split=SPLIT, n_seq=len(seqs), hover_threshold_px=thr, hover_threshold_source="results/p0/hover_threshold.json (TRAIN, đóng băng)",
               ego_counts=pd.Series(ego).value_counts().to_dict(),
               n_frames_images=int(tab.n_img.sum()), n_frames_used=int(tab.n_frames_used.sum()), n_frames_dropped=int(tab.n_frames_dropped.sum()),
               resolutions=tab.apply(lambda r: f"{r.W}x{r.H}", axis=1).value_counts().to_dict(),
               n_E3_main=int(n_all), n_E3_border=int(e3.border_entry.sum()), n_E3_interior=int((~e3.border_entry.astype(bool)).sum()),
               n_E3_birth_window=int(len(e3w)), strata=strata,
               M_bin_counts=e3w.M_bin.value_counts().to_dict(), sequences=tab.to_dict("records"))
    (OUT / "test_prepare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "sequences"}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
