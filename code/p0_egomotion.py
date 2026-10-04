"""P0 bước 5: ego-motion (homography nền) + proxy nhiễu cue + chi phí CPU.

Mỗi chuỗi: cặp (t, t+1) mỗi 10 khung, tối đa 300 cặp. ORB 2000 điểm (che bbox GT mọi loại, kể cả
vùng ignore), BF Hamming + ratio 0.75, findHomography RANSAC 3 px.
Ghi: dịch chuyển tâm ảnh qua H (px/khung), góc xoay (độ), đổi tỉ lệ, tỉ lệ inlier;
proxy nhiễu: tỉ lệ pixel NGOÀI bbox có |diff| > 25 (xám) — thô và sau warp bằng H;
chi phí: ms/cặp cho ORB+match+RANSAC và cho warp+diff.
Xuất results/p0/egomotion_samples.parquet, egomotion_seq.parquet, egomotion_summary.json;
gộp khối "egomotion" vào results/p0_stats.json.
Chạy: python code/p0_egomotion.py [uavdt|visdrone|all] [train|test|all]   (split mặc định all; P2E: "uavdt train" — TEST cấm)
"""
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import UAVDT_ATTR as UAVDT_ATTR_DIR, UAVDT_FRAMES, UAVDT_GT, visdrone_split  # noqa: E402
P0 = ROOT / "results" / "p0"
STATS = ROOT / "results" / "p0_stats.json"
STEP, MAX_PAIRS, DIFF_THR = 10, 300, 25
HOVER_THR_PX = 1.0  # đề xuất; anh Đạt chốt ở CP0
cv2.setNumThreads(1)  # đo chi phí đơn luồng, ổn định


def seq_sources(tag, only_split="all"):
    """(seq, split, [img paths], {frame: [(x,y,w,h),...]}) cho mọi chuỗi có ảnh."""
    out = []
    if tag == "uavdt":
        gt = UAVDT_GT / "GT"
        img_root = UAVDT_FRAMES
        split = {p.name.split("_")[0].strip(): sp for sp in ("train", "test")
                 for p in (UAVDT_ATTR_DIR / "M_attr" / sp).glob("*_attr.txt")}
        for p in sorted(gt.glob("*_gt_whole.txt")):
            seq = p.name.split("_")[0]
            if only_split != "all" and split.get(seq) != only_split:
                continue  # lọc TRƯỚC khi đọc ảnh/GT (P2E: TEST cấm)
            imgs = sorted((img_root / seq).glob("*.jpg"))
            if not imgs:
                continue
            boxes = {}
            for f in (p, gt / f"{seq}_gt_ignore.txt"):
                if f.exists() and f.stat().st_size > 0:  # gt_ignore có thể rỗng
                    a = pd.read_csv(f, header=None).iloc[:, :6].values
                    for fr, _, x, y, w, h in a:
                        boxes.setdefault(int(fr), []).append((x, y, w, h))
            out.append((seq, split.get(seq, "?"), imgs[:max(boxes)] if boxes else imgs, boxes))  # M0207: ảnh > GT → cắt về phạm vi GT
    else:
        for sp in ("train", "val", "test-dev"):
            d = visdrone_split(sp)
            for p in sorted((d / "annotations").glob("*.txt")) if d.exists() else []:
                imgs = sorted((d / "sequences" / p.stem).glob("*.jpg"))
                if not imgs:
                    continue
                a = pd.read_csv(p, header=None).iloc[:, :6].values  # mọi box kể cả ignored/others
                boxes = {}
                for fr, _, x, y, w, h in a:
                    boxes.setdefault(int(fr), []).append((x, y, w, h))
                out.append((p.stem, sp, imgs, boxes))
    return out


def bbox_mask(shape, boxes, pad=2):
    m = np.full(shape, 255, np.uint8)
    H, W = shape
    for x, y, w, h in boxes:
        x0, y0 = max(int(x) - pad, 0), max(int(y) - pad, 0)
        x1, y1 = min(int(x + w) + pad, W), min(int(y + h) + pad, H)
        m[y0:y1, x0:x1] = 0
    return m


def process_seq(tag, seq, split, imgs, boxes, orb, bf):
    recs = []
    n = len(imgs)
    ts = list(range(0, n - 1, STEP))[:MAX_PAIRS]
    for t in ts:
        f0, f1 = t + 1, t + 2  # frame index 1-based
        g0 = cv2.imread(str(imgs[t]), cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(str(imgs[t + 1]), cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None or g0.shape != g1.shape:
            continue
        H_, W_ = g0.shape
        m0 = bbox_mask(g0.shape, boxes.get(f0, []))
        m1 = bbox_mask(g0.shape, boxes.get(f1, []))
        t_a = time.perf_counter()
        k0, d0 = orb.detectAndCompute(g0, m0)
        k1, d1 = orb.detectAndCompute(g1, m1)
        good = []
        if d0 is not None and d1 is not None and len(k0) >= 2 and len(k1) >= 2:
            for pair in bf.knnMatch(d0, d1, k=2):
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                    good.append(pair[0])
        Hm, inl = None, None
        if len(good) >= 4:
            src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        t_b = time.perf_counter()
        rec = dict(dataset="UAVDT" if tag == "uavdt" else "VisDrone", split=split, seq=seq, frame=f0, W=W_, H=H_,
                   n_kp0=len(k0), n_good=len(good), ms_orb_ransac=(t_b - t_a) * 1e3)
        outside = (m0 > 0) & (m1 > 0)
        raw = cv2.absdiff(g0, g1) > DIFF_THR
        rec["diff_raw"] = float(raw[outside].mean()) if outside.any() else np.nan
        if Hm is not None:
            c = np.array([[[W_ / 2, H_ / 2]]], np.float32)
            c2 = cv2.perspectiveTransform(c, Hm)[0, 0]
            A = Hm[:2, :2] / Hm[2, 2]
            rec.update(shift_px=float(np.hypot(*(c2 - c[0, 0]))), rot_deg=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                       scale=float(np.sqrt(abs(np.linalg.det(A)))), inlier_ratio=float(inl.sum() / len(good)), H_ok=True)
            t_c = time.perf_counter()
            warped = cv2.warpPerspective(g0, Hm, (W_, H_))
            valid = cv2.warpPerspective(np.full_like(g0, 255), Hm, (W_, H_)) > 0
            wd = cv2.absdiff(warped, g1) > DIFF_THR
            t_d = time.perf_counter()
            ok = outside & valid
            rec["diff_warp"] = float(wd[ok].mean()) if ok.any() else np.nan
            rec["ms_warp_diff"] = (t_d - t_c) * 1e3
        else:
            rec.update(shift_px=np.nan, rot_deg=np.nan, scale=np.nan, inlier_ratio=np.nan, H_ok=False,
                       diff_warp=np.nan, ms_warp_diff=np.nan)
        recs.append(rec)
    return recs


def main(which, split="all"):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_recs = []
    old = P0 / "egomotion_samples.parquet"
    tags = ("uavdt", "visdrone") if which == "all" else (which,)
    if old.exists():  # giữ kết quả của dataset không chạy lại
        prev = pd.read_parquet(old)
        keep = [t for t in ("uavdt", "visdrone") if t not in tags]
        all_recs.append(prev[prev.dataset.isin(["UAVDT" if t == "uavdt" else "VisDrone" for t in keep])])
    for tag in tags:
        srcs = [x for x in seq_sources(tag, split) if split == "all" or x[1] == split]
        print(tag, "seqs with images:", len(srcs))
        recs = []
        for i, (seq, split, imgs, boxes) in enumerate(srcs):
            recs += process_seq(tag, seq, split, imgs, boxes, orb, bf)
            print(f"  [{i + 1}/{len(srcs)}] {seq} pairs={len(recs)}", flush=True)
        if recs:
            all_recs.append(pd.DataFrame(recs))
    s = pd.concat([d for d in all_recs if len(d)], ignore_index=True) if all_recs else pd.DataFrame()
    if s.empty:
        print("no frames available")
        return
    s.to_parquet(P0 / "egomotion_samples.parquet", index=False)
    seq = s.groupby(["dataset", "split", "seq"]).agg(
        n_pairs=("frame", "size"), H_ok_rate=("H_ok", "mean"), shift_px_median=("shift_px", "median"),
        rot_deg_median=("rot_deg", lambda x: float(np.nanmedian(np.abs(x)))), scale_median=("scale", "median"),
        inlier_ratio_median=("inlier_ratio", "median"), diff_raw_median=("diff_raw", "median"),
        diff_warp_median=("diff_warp", "median"), W=("W", "first"), H=("H", "first")).reset_index()
    seq["ego"] = np.where(seq.shift_px_median < HOVER_THR_PX, "hovering", "moving")
    seq.to_parquet(P0 / "egomotion_seq.parquet", index=False)
    summ = {"split": split, "hover_threshold_px": HOVER_THR_PX, "cpu": platform.processor(), "cv2": cv2.__version__,
            "threads": 1, "step": STEP, "max_pairs": MAX_PAIRS, "diff_thr": DIFF_THR, "by_dataset": {}}
    for ds, d in seq.groupby("dataset"):
        ss = s[s.dataset == ds]
        summ["by_dataset"][ds] = dict(
            n_seq=int(len(d)), n_pairs=int(len(ss)), H_ok_rate=float(ss.H_ok.mean()),
            n_seq_hovering=int((d.ego == "hovering").sum()), share_seq_hovering=float((d.ego == "hovering").mean()),
            shift_seq_median_quantiles={f"p{p}": float(np.nanpercentile(d.shift_px_median, p)) for p in (10, 25, 50, 75, 90)},
            n_seq_shift_lt={str(t): int((d.shift_px_median < t).sum()) for t in (0.5, 1.0, 2.0, 5.0)},
            rot_deg_median=float(d.rot_deg_median.median()), scale_dev_median=float((d.scale_median - 1).abs().median()),
            inlier_ratio_median=float(d.inlier_ratio_median.median()),
            diff_raw_median=float(d.diff_raw_median.median()), diff_warp_median=float(d.diff_warp_median.median()),
            diff_by_ego={e: dict(raw=float(g.diff_raw_median.median()), warp=float(g.diff_warp_median.median()), n_seq=int(len(g)))
                         for e, g in d.groupby("ego")},
            ms_orb_ransac_median=float(ss.ms_orb_ransac.median()), ms_orb_ransac_p90=float(ss.ms_orb_ransac.quantile(.9)),
            ms_warp_diff_median=float(ss.ms_warp_diff.median()), ms_warp_diff_p90=float(ss.ms_warp_diff.quantile(.9)),
            n_timing_pairs=int(ss.ms_warp_diff.notna().sum()),
            resolutions=d.apply(lambda r: f"{r.W}x{r.H}", axis=1).value_counts().to_dict())
    (P0 / "egomotion_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}
    stats["egomotion"] = summ
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all", sys.argv[2] if len(sys.argv) > 2 else "all")
