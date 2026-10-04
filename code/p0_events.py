"""P0 bước 3: định nghĩa sự kiện E0–E3 từ rows_/tracks_/frames_<ds>.parquet.

VISIBLE (chính) / VISIBLE_STRICT (độ nhạy) đã gắn ở p0_parse.py.
  E0  class-agnostic: khung có ≥1 VEHICLE visible → rho_ev mỗi chuỗi (events_E0.parquet, 1 dòng/chuỗi)
  E1  vòng đời track (lifetime = last-first+1)
  E2  mỗi đoạn liên tiếp VISIBLE (khung liền kề, có annotation) của một track = 1 sự kiện, độ dài D
  E3  track VEHICLE born_inside=True; D = đoạn E2 đầu tiên; tách border / interior; M(t) tại khung sinh
Mỗi bảng có cột vis_def ∈ {main, strict} (E1 không phụ thuộc định nghĩa visible → chỉ main).
P0b (A1): E2/E3 có thêm seq_first, seq_last, right_censored (visible ở khung cuối chuỗi), left_censored (bắt đầu ở khung đầu).
P2G (Q5, 01-10-2026): KHÔNG lọc E3 theo audit khung trước sinh (luật loại annotation-late của P2F bị rút; audit chỉ mô tả,
  code/p2/p2_audit_prebirth.py đọc events_E3_<ds>.parquet). E3 = định nghĩa P0 đầy đủ.
Chạy: python code/p0_events.py [uavdt|visdrone|all]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p0"


def visible_runs(frames, vis):
    """Trả list (start, end) các đoạn liên tiếp: khung liền kề (Δ=1) và vis=True."""
    runs = []
    start = prev = None
    for f, v in zip(frames, vis):
        if v and start is not None and f == prev + 1:
            prev = f
            continue
        if start is not None:
            runs.append((start, prev))
            start = None
        if v:
            start = prev = f
    if start is not None:
        runs.append((start, prev))
    return runs


def events_for(tag):
    rows = pd.read_parquet(OUT / f"rows_{tag}.parquet")
    tracks = pd.read_parquet(OUT / f"tracks_{tag}.parquet")
    frames = pd.read_parquet(OUT / f"frames_{tag}.parquet")
    attr_cols = [c for c in ("altitude", "view", "light", "long_term") if c in tracks.columns]
    tinfo = tracks.set_index(["seq", "track_id"])
    Mlookup = {vd: frames.set_index(["seq", "frame"])[col] for vd, col in (("main", "M"), ("strict", "M_strict"))}
    rows = rows.sort_values(["seq", "tid", "frame"])

    # E0
    e0 = []
    for vd, col in (("main", "M"), ("strict", "M_strict")):
        g = frames.groupby(["dataset", "split", "seq"])
        d = g.agg(n_frames=(col, "size"), n_ev_frames=(col, lambda x: int((x > 0).sum())),
                  M_mean=(col, "mean"), M_median=(col, "median"), M_max=(col, "max")).reset_index()
        d["rho_ev"] = d.n_ev_frames / d.n_frames
        d["vis_def"] = vd
        e0.append(d)
    e0 = pd.concat(e0, ignore_index=True)

    # E1
    e1 = tracks[["dataset", "split", "seq", "track_id", "group", "category", "first", "lifetime",
                 "sqrt_area_median", "speed_px_median", "speed_norm", "born_inside", "border_entry"] + attr_cols].copy()
    e1 = e1.rename(columns={"first": "start", "lifetime": "D"})
    e1["M_at_start"] = [Mlookup["main"].get((s, f), 0) for s, f in zip(e1.seq, e1.start)]
    e1["vis_def"] = "main"

    # E2 / E3
    e2_recs, e3_recs = [], []
    for (seq, tid), d in rows.groupby(["seq", "tid"], sort=False):
        t = tinfo.loc[(seq, tid)]
        fr = d.frame.values
        sa = d.sqrt_area.values
        spd = d.speed_px.values
        for vd, col in (("main", "visible"), ("strict", "visible_strict")):
            runs = visible_runs(fr, d[col].values)
            for i, (a, b) in enumerate(runs):
                m = (fr >= a) & (fr <= b)
                sp = spd[m][1:]  # tốc độ trong đoạn (bỏ khung đầu vì Δ tính với khung trước đoạn)
                rec = dict(dataset=t.dataset, split=t.split, seq=seq, track_id=int(tid), group=t.group,
                           category=t.category, run_idx=i, start=int(a), D=int(b - a + 1),
                           sqrt_area_median=float(np.median(sa[m])),
                           speed_px_median=float(np.nanmedian(sp)) if np.isfinite(sp).any() else float(t.speed_px_median),
                           M_at_start=int(Mlookup[vd].get((seq, a), 0)), vis_def=vd,
                           born_inside=bool(t.born_inside), border_entry=bool(t.border_entry))
                for c in attr_cols:
                    rec[c] = t[c]
                e2_recs.append(rec)
                if i == 0 and t.group == "VEHICLE" and t.born_inside:
                    r3 = dict(rec)
                    r3["birth_frame"] = int(t["first"])
                    r3["onset_delay"] = int(a - t["first"])  # khung từ lúc sinh tới lúc visible đầu tiên
                    r3["M_at_birth"] = int(Mlookup[vd].get((seq, int(t["first"])), 0))
                    r3["e3_type"] = "border" if t.border_entry else "interior"
                    e3_recs.append(r3)
            if not runs and t.group == "VEHICLE" and t.born_inside:
                e3_recs.append(dict(dataset=t.dataset, split=t.split, seq=seq, track_id=int(tid), group=t.group,
                                    category=t.category, run_idx=-1, start=-1, D=0, vis_def=vd,
                                    born_inside=True, border_entry=bool(t.border_entry), birth_frame=int(t["first"]),
                                    e3_type="border" if t.border_entry else "interior", never_visible=True,
                                    **{c: t[c] for c in attr_cols}))
    e2 = pd.DataFrame(e2_recs)
    e3 = pd.DataFrame(e3_recs)
    if "never_visible" not in e3.columns:
        e3["never_visible"] = False
    e3["never_visible"] = e3["never_visible"].fillna(False).astype(bool)

    # P0b-A1: cờ kiểm duyệt. right_censored = còn visible ở khung cuối chuỗi (D bị cắt);
    # left_censored (E2) = bắt đầu ở khung đầu chuỗi (thời điểm sinh thật không quan sát được).
    first_f = frames.groupby("seq").frame.min()
    last_f = frames.groupby("seq").frame.max()
    for d in (e2, e3):
        d["seq_first"] = d.seq.map(first_f).astype(int)
        d["seq_last"] = d.seq.map(last_f).astype(int)
        ok = d.start >= 0
        d["right_censored"] = ok & (d.start + d.D - 1 >= d.seq_last)
        d["left_censored"] = ok & (d.start <= d.seq_first)

    # tỉ lệ đến E3 theo 100 khung (mỗi chuỗi)
    nfr = frames.groupby("seq").size()
    arr = (e3[~e3.never_visible].groupby(["vis_def", "seq"]).size().rename("n_E3").reset_index())
    arr["n_frames"] = arr.seq.map(nfr)
    arr["arrival_per_100f"] = 100 * arr.n_E3 / arr.n_frames

    e0.to_parquet(OUT / f"events_E0_{tag}.parquet", index=False)
    e1.to_parquet(OUT / f"events_E1_{tag}.parquet", index=False)
    e2.to_parquet(OUT / f"events_E2_{tag}.parquet", index=False)
    e3.to_parquet(OUT / f"events_E3_{tag}.parquet", index=False)
    arr.to_parquet(OUT / f"events_E3_arrival_{tag}.parquet", index=False)
    summ = {
        "E1_n": int(len(e1)), "E2_n": e2.groupby(["vis_def", "group"]).size().to_dict(),
        "E3_n": e3[~e3.never_visible].groupby(["vis_def", "e3_type"]).size().to_dict(),
        "E3_never_visible": e3[e3.never_visible].groupby("vis_def").size().to_dict(),
        "E3_right_censored": e3[~e3.never_visible].groupby("vis_def").right_censored.sum().to_dict(),
        "E2_right_censored": e2.groupby("vis_def").right_censored.sum().to_dict(),
        "E2_left_censored": e2.groupby("vis_def").left_censored.sum().to_dict(),
    }
    print(tag, summ)
    return summ


def merge_all():
    """Gộp các dataset → events_E1/E2/E3.parquet (đầu ra theo tên trong prompt)."""
    for e in ("E0", "E1", "E2", "E3", "E3_arrival"):
        parts = [pd.read_parquet(p) for p in sorted(OUT.glob(f"events_{e}_*.parquet"))
                 if p.stem.replace(f"events_{e}_", "") in ("uavdt", "visdrone")]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(OUT / f"events_{e}.parquet", index=False)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for tag in ("uavdt", "visdrone"):
        if which in (tag, "all") and (OUT / f"rows_{tag}.parquet").exists():
            events_for(tag)
    merge_all()
