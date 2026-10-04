"""P0 bước 2: parse annotation VisDrone2019-VID / UAVDT-MOT → bảng dòng, bảng track, bảng khung.

Xuất (results/p0/):
  rows_<ds>.parquet    1 dòng/box sạch (đã lọc), có cờ visible / visible_strict
  tracks_<ds>.parquet  1 dòng/track
  frames_<ds>.parquet  1 dòng/khung: M(t) = số VEHICLE visible
  parse_<ds>.json      số đếm (clip, khung, box bị lọc, kích thước ảnh, độ lệch so kỳ vọng)
Chạy: python code/p0_parse.py [uavdt|visdrone|all]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import UAVDT_ATTR as UAVDT_ATTR_DIR, UAVDT_FRAMES, UAVDT_GT, visdrone_split  # noqa: E402
OUT = ROOT / "results" / "p0"
OUT.mkdir(parents=True, exist_ok=True)

BORDER_FRAC = 0.02

VD_VEHICLE = {3: "bicycle", 4: "car", 5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor"}
VD_PERSON = {1: "pedestrian", 2: "people"}
UAVDT_CAT = {1: "car", 2: "truck", 3: "bus"}
UAVDT_ATTR = ["daylight", "night", "fog", "low_alt", "medium_alt", "high_alt", "front_view", "side_view", "bird_view", "long_term"]


def gt_frame_range(n_img, gt_max):
    """PREREG §1 (P2F): số khung dùng = min(số ảnh, khung GT cuối) — khung ngoài GT không có nhãn (không phải "0 xe").
    Ví dụ M0207: 885 ảnh, GT tới 571 → 571. Không có ảnh (chưa tải frames) → khung GT cuối."""
    return min(int(n_img), int(gt_max)) if n_img else int(gt_max)


def is_border(x, y, w, h, W, H, frac=BORDER_FRAC):
    """Box chạm dải biên rộng frac × kích thước ảnh THẬT (PREREG §1: M0901 rộng 960 → biên phải tính theo 960)."""
    return bool((x <= frac * W) or (y <= frac * H) or (x + w >= (1 - frac) * W) or (y + h >= (1 - frac) * H))


def image_size(img_dir):
    """Đọc kích thước thật từ khung đầu (không giả định)."""
    import cv2
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")) if img_dir.exists() else []
    if not imgs:
        return None, 0
    im = cv2.imread(str(imgs[0]))
    return (int(im.shape[1]), int(im.shape[0])), len(imgs)


def runs_count(frames_sorted):
    """Số đoạn hở (gap run) giữa first..last."""
    d = np.diff(frames_sorted)
    return int((d > 1).sum()), int((d - 1)[d > 1].sum())


def build_tracks(rows, seq_meta):
    """rows: dataset, split, seq, frame, tid, x,y,w,h, group, category, visible...; seq_meta: seq -> dict(W,H,first_frame,...)"""
    rows = rows.sort_values(["seq", "tid", "frame"]).reset_index(drop=True)
    rows["cx"] = rows.x + rows.w / 2
    rows["cy"] = rows.y + rows.h / 2
    rows["sqrt_area"] = np.sqrt(rows.w * rows.h)
    g = rows.groupby(["seq", "tid"], sort=False)
    # tốc độ chỉ trên cặp khung liền kề (Δframe == 1) của cùng track
    same = (rows.seq.values[1:] == rows.seq.values[:-1]) & (rows.tid.values[1:] == rows.tid.values[:-1])
    adj = same & (np.diff(rows.frame.values) == 1)
    sp = np.full(len(rows), np.nan)
    sp[1:][adj] = np.hypot(np.diff(rows.cx.values), np.diff(rows.cy.values))[adj]
    rows["speed_px"] = sp
    recs = []
    for (seq, tid), d in g:
        m = seq_meta[seq]
        W, H = m["W"], m["H"]
        fr = d.frame.values
        n_gap_runs, gap_frames = runs_count(fr)
        f0 = d.iloc[0]
        be = is_border(f0.x, f0.y, f0.w, f0.h, W, H)
        spd = np.nanmedian(d.speed_px.values) if np.isfinite(d.speed_px.values).any() else np.nan
        rec = dict(dataset=f0.dataset, split=f0.split, seq=seq, track_id=int(tid), group=f0.group,
                   category=d.category.mode().iloc[0], first=int(fr[0]), last=int(fr[-1]), n_annot=len(fr),
                   lifetime=int(fr[-1] - fr[0] + 1), gaps=n_gap_runs, gap_frames=gap_frames,
                   sqrt_area_median=float(np.median(d.sqrt_area)), speed_px_median=float(spd),
                   speed_norm=float(spd / np.hypot(W, H)) if np.isfinite(spd) else np.nan,
                   born_inside=bool(fr[0] > m["first_frame"]), border_entry=bool(be),
                   n_visible=int(d.visible.sum()), n_visible_strict=int(d.visible_strict.sum()))
        for k, v in m.get("attrs", {}).items():
            rec[k] = v
        recs.append(rec)
    return rows, pd.DataFrame(recs)


def build_frames(rows, seq_meta):
    out = []
    veh = rows[rows.group == "VEHICLE"]
    per = rows[rows.group == "PERSON"]
    for seq, m in seq_meta.items():
        n = m["n_frames"]
        idx = np.arange(m["first_frame"], m["first_frame"] + n)
        def cnt(df, col=None):
            d = df[df.seq == seq]
            if col is not None:
                d = d[d[col]]
            return d.groupby("frame").size().reindex(idx, fill_value=0).values
        f = pd.DataFrame(dict(dataset=m["dataset"], split=m["split"], seq=seq, frame=idx,
                              M=cnt(veh, "visible"), M_strict=cnt(veh, "visible_strict"), M_all=cnt(veh),
                              P=cnt(per, "visible")))
        for k, v in m.get("attrs", {}).items():
            f[k] = v
        out.append(f)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- UAVDT
def parse_uavdt():
    base = UAVDT_GT / "GT"
    attr_dir = UAVDT_ATTR_DIR / "M_attr"
    img_root = UAVDT_FRAMES
    split_of = {}
    attrs = {}
    for sp in ("train", "test"):
        for p in sorted((attr_dir / sp).glob("*_attr.txt")):
            s = p.name.split("_")[0].strip()  # dataset có file "M0701 _attr.txt" (dấu cách thừa)
            split_of[s] = sp
            v = [int(x) for x in p.read_text().strip().split(",")]
            a = dict(zip(UAVDT_ATTR, v))
            a["altitude"] = "low" if a["low_alt"] else "medium" if a["medium_alt"] else "high" if a["high_alt"] else "?"
            a["view"] = "front" if a["front_view"] else "side" if a["side_view"] else "bird" if a["bird_view"] else "?"
            a["light"] = "daylight" if a["daylight"] else "night" if a["night"] else "fog" if a["fog"] else "?"
            attrs[s] = a
    info = {"dataset": "UAVDT", "notes": []}
    all_rows, seq_meta = [], {}
    gt_files = sorted(base.glob("*_gt_whole.txt"))
    dup_total = 0
    for p in gt_files:
        seq = p.name.split("_")[0]
        a = pd.read_csv(p, header=None, names=["frame", "tid", "x", "y", "w", "h", "out_of_view", "occlusion", "cat"])
        dups = a.duplicated(["frame", "tid"]).sum()
        dup_total += int(dups)
        a = a.drop_duplicates(["frame", "tid"])
        sz, n_img = image_size(img_root / seq)
        gt_ext = (int((a.x + a.w).max()), int((a.y + a.h).max()))
        if sz is None:
            # TẠM khi chưa có frames: GT extent toàn bộ dataset = 1025x541 → 1024x540, lệch với paper (1080x540).
            # Thay bằng kích thước thật đọc từ ảnh khi có frames — xem parse_uavdt.json.
            W, H = 1024, 540
            size_src = "gt_extent_provisional"
        else:
            (W, H), size_src = sz, "image"
        # P2E-D1: M0207 có 885 ảnh nhưng GT chỉ 1..571 → khung ngoài GT không có nhãn (không phải "0 xe") → chỉ lấy phạm vi GT
        n_frames = gt_frame_range(n_img, a.frame.max())
        if n_img > int(a.frame.max()):
            info["notes"].append(f"{seq}: {n_img} ảnh > GT max frame {int(a.frame.max())} → n_frames = phạm vi GT")
        seq_meta[seq] = dict(dataset="UAVDT", split=split_of.get(seq, "?"), W=W, H=H, size_src=size_src,
                             gt_extent=gt_ext, n_frames=n_frames, n_img=n_img, first_frame=1,
                             gt_max_frame=int(a.frame.max()), gt_min_frame=int(a.frame.min()),
                             attrs={k: attrs.get(seq, {}).get(k) for k in ("altitude", "view", "light", "long_term")})
        a["dataset"], a["split"], a["seq"] = "UAVDT", split_of.get(seq, "?"), seq
        a["group"] = "VEHICLE"
        a["category"] = a.cat.map(UAVDT_CAT).fillna("unk")
        a["visible"] = (a.occlusion != 2) & (a.out_of_view != 2)
        a["visible_strict"] = (a.occlusion == 1) & (a.out_of_view == 1)
        all_rows.append(a)
    rows = pd.concat(all_rows, ignore_index=True)
    info.update(
        n_seq=len(seq_meta), n_seq_by_split={s: sum(1 for m in seq_meta.values() if m["split"] == s) for s in ("train", "test", "?")},
        n_frames_total=int(sum(m["n_frames"] for m in seq_meta.values())),
        n_frames_by_split={s: int(sum(m["n_frames"] for m in seq_meta.values() if m["split"] == s)) for s in ("train", "test")},
        frames_source="images" if all(m["n_img"] for m in seq_meta.values()) else "gt_max_frame",
        n_boxes=len(rows), dup_frame_tid_dropped=dup_total,
        category_counts=rows.category.value_counts().to_dict(),
        occlusion_counts={int(k): int(v) for k, v in rows.occlusion.value_counts().items()},
        out_of_view_counts={int(k): int(v) for k, v in rows.out_of_view.value_counts().items()},
        image_sizes=sorted({f"{m['W']}x{m['H']}({m['size_src']})" for m in seq_meta.values()}),
        gt_extent_max=[int(max(m["gt_extent"][0] for m in seq_meta.values())), int(max(m["gt_extent"][1] for m in seq_meta.values()))],
        seq_len_min=int(min(m["n_frames"] for m in seq_meta.values())), seq_len_max=int(max(m["n_frames"] for m in seq_meta.values())),
        altitude_counts=pd.Series([m["attrs"]["altitude"] for m in seq_meta.values()]).value_counts().to_dict(),
        seq_meta=seq_meta,
    )
    return finish("uavdt", rows, seq_meta, info)


# ---------------------------------------------------------------- VisDrone
def parse_visdrone():
    info = {"dataset": "VisDrone2019-VID", "notes": []}
    all_rows, seq_meta = [], {}
    dropped = {"score0": 0, "cat0_ignored": 0, "cat11_others": 0, "dup": 0}
    frames_annot_total = 0
    for split in ("train", "val", "test-dev"):
        d = visdrone_split(split)
        if not d.exists():
            info["notes"].append(f"MISSING split {split}: {d}")
            continue
        for p in sorted((d / "annotations").glob("*.txt")):
            seq = p.stem
            a = pd.read_csv(p, header=None)
            a = a.iloc[:, :10]
            a.columns = ["frame", "tid", "x", "y", "w", "h", "score", "cat", "truncation", "occlusion"]
            frames_annot_total += a.frame.nunique()
            dropped["score0"] += int((a.score == 0).sum())
            dropped["cat0_ignored"] += int((a.cat == 0).sum())
            dropped["cat11_others"] += int((a.cat == 11).sum())
            a = a[(a.score != 0) & (a.cat != 0) & (a.cat != 11)]
            dropped["dup"] += int(a.duplicated(["frame", "tid"]).sum())
            a = a.drop_duplicates(["frame", "tid"])
            sz, n_img = image_size(d / "sequences" / seq)
            W, H = sz if sz else (None, None)
            seq_meta[seq] = dict(dataset="VisDrone", split=split, W=W, H=H, size_src="image" if sz else "missing",
                                 n_frames=n_img, n_img=n_img, first_frame=1, attrs={},
                                 gt_max_frame=int(a.frame.max()), gt_min_frame=int(a.frame.min()))
            a["dataset"], a["split"], a["seq"] = "VisDrone", split, seq
            a["group"] = np.where(a.cat.isin(list(VD_VEHICLE)), "VEHICLE", np.where(a.cat.isin(list(VD_PERSON)), "PERSON", "OTHER"))
            a["category"] = a.cat.map({**VD_VEHICLE, **VD_PERSON}).fillna("unk")
            a["visible"] = a.occlusion < 2
            a["visible_strict"] = (a.occlusion == 0) & (a.truncation == 0)
            all_rows.append(a)
    if not all_rows:
        info["status"] = "NO_DATA"
        (OUT / "parse_visdrone.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
        print("VisDrone: không có dữ liệu")
        return None
    rows = pd.concat(all_rows, ignore_index=True)
    info.update(
        n_seq=len(seq_meta), n_seq_by_split={s: sum(1 for m in seq_meta.values() if m["split"] == s) for s in ("train", "val", "test-dev")},
        n_frames_total=int(sum(m["n_frames"] for m in seq_meta.values())),
        n_frames_by_split={s: int(sum(m["n_frames"] for m in seq_meta.values() if m["split"] == s)) for s in ("train", "val", "test-dev")},
        n_frames_with_annotation=int(frames_annot_total), dropped=dropped, n_boxes=len(rows),
        category_counts=rows.category.value_counts().to_dict(),
        occlusion_counts={int(k): int(v) for k, v in rows.occlusion.value_counts().items()},
        truncation_counts={int(k): int(v) for k, v in rows.truncation.value_counts().items()},
        image_sizes=pd.Series([f"{m['W']}x{m['H']}" for m in seq_meta.values()]).value_counts().to_dict(),
        seq_len_min=int(min(m["n_frames"] for m in seq_meta.values())), seq_len_max=int(max(m["n_frames"] for m in seq_meta.values())),
        seq_meta=seq_meta,
    )
    return finish("visdrone", rows, seq_meta, info)


def finish(tag, rows, seq_meta, info):
    rows, tracks = build_tracks(rows, seq_meta)
    frames = build_frames(rows, seq_meta)
    keep = ["dataset", "split", "seq", "frame", "tid", "x", "y", "w", "h", "group", "category", "visible", "visible_strict", "sqrt_area", "speed_px", "cx", "cy"]
    keep += [c for c in ("occlusion", "out_of_view") if c in rows.columns]  # P2G: audit pre-birth phân loại A1/A2 cần cờ GT gốc
    rows[keep].to_parquet(OUT / f"rows_{tag}.parquet", index=False)
    tracks.to_parquet(OUT / f"tracks_{tag}.parquet", index=False)
    frames.to_parquet(OUT / f"frames_{tag}.parquet", index=False)
    info["n_tracks"] = int(len(tracks))
    info["n_tracks_by_group"] = tracks.group.value_counts().to_dict()
    info["status"] = "OK"
    (OUT / f"parse_{tag}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(tag, {k: v for k, v in info.items() if k != "seq_meta"})
    return info


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("uavdt", "all"):
        parse_uavdt()
    if which in ("visdrone", "all"):
        parse_visdrone()
