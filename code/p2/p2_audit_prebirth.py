"""P2G (Q5): audit khung trước onset của sự kiện E3 — CHỈ MÔ TẢ, KHÔNG loại sự kiện (luật loại của P2F-Q4 bị rút 01-10-2026).

Sự kiện: E3 (VEHICLE born_inside, VISIBLE chính: occlusion ≠ 2 và out_of_view ≠ 2), b = khung onset = khung VISIBLE đầu tiên
  (start — đúng onset của KPI độ trễ). P2F dùng b = khung GT đầu tiên của track (birth_frame); khi đó track không thể có dòng GT
  trước b nên trường hợp (ii) "đã có trong GT nhưng occlusion/out_of_view = 2" bị gộp lẫn vào "gán trễ" — đó là lý do rút luật.
Detector: dump yolo26s@1024, score ≥ 0,25 (mọi lớp, như recall), 5 khung b−5 … b−1; khớp = IoU ≥ 0,3 với box GT của track ở khung b.
Nhãn (luật cố định PREREG §5, áp cho TEST sau freeze):
  censored-start         b − 5 < khung đầu chuỗi
  true-birth             ≤ 2/5 khung khớp
  với ≥ 3/5 khung khớp:
  partial-entry (A1)     border VÀ (box GT khung b cách mép ảnh THẬT ≤ 2 px HOẶC track có dòng GT ở b−5…b−1 với out_of_view = 2)
  visibility-transition (A2)  track có dòng GT ở b−5…b−1 với occlusion = 2 (hoặc out_of_view = 2 khi interior)
  annotation-late (A3)   KHÔNG có dòng GT nào của track trong b−5…b−1 (ID mới) mà detector đã thấy ≥ 3/5 — "gán trễ" THẬT
  true-birth (A4)        còn lại
Không loại sự kiện nào. A3 > 20 % interior (TRAIN) → ghi hạn chế ở PREREG §6 (vẫn không loại).
Độ nhạy (phụ lục, mô tả): onset_alt = khung sớm nhất trong b−5…b−1 có detector khớp (không có → b); shift = b − onset_alt.
Placebo: box cùng cỡ đặt ngẫu nhiên (seed 42) ở khung b−3 → tỉ lệ khớp ngẫu nhiên.
Kiểm phụ (MODEL-ASSISTED, không phải nhãn người): results/p0/audit_short_visual_<split>.csv cột visual_v2.
Xuất: results/p0/audit_prebirth_<split>.json.  Chạy: python code/p2/p2_audit_prebirth.py [--split train]
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
from p2_match import iou_matrix  # noqa: E402
from paths import DUMPS_DIR  # noqa: E402

P0 = ROOT / "results" / "p0"
K_PRE, K_LATE, IOU_MIN, SCORE_MIN, EDGE_PX = 5, 3, 0.3, 0.25, 2
EDGE_PX_SENS = 5  # P2H-Q9: độ nhạy ngưỡng mép (chỉ mô tả)
DUMP = "yolo26s_1024"
LABELS = ["partial-entry", "visibility-transition", "annotation-late", "true-birth", "censored-start"]
POLICY = "descriptive"  # P2G-Q5: audit không loại sự kiện


def det_matches(gt_box, dets, b, lo, k_pre=K_PRE, iou_min=IOU_MIN, score_min=SCORE_MIN):
    """IoU lớn nhất theo khung f ∈ [max(lo, b−k_pre), b) giữa box tham chiếu và detection score ≥ score_min. Trả {f: iou}."""
    ref = np.asarray(gt_box, float).reshape(1, 4)
    d = dets[(dets.frame >= b - k_pre) & (dets.frame < b) & (dets.score >= score_min)]
    out = {}
    for f in range(max(lo, b - k_pre), b):
        df = d[d.frame == f]
        out[f] = float(iou_matrix(ref, df[["x", "y", "w", "h"]].values.astype(float)).max()) if len(df) else 0.0
    return out


def classify(n_match, censored, e3_type, edge_px, prior, edge_max=EDGE_PX):
    """prior: DataFrame dòng GT của CÙNG track ở b−5…b−1 (cột occlusion, out_of_view). Trả nhãn (luật PREREG §5)."""
    if censored:
        return "censored-start"
    if n_match < K_LATE:
        return "true-birth"
    oov2 = bool((prior.out_of_view == 2).any()) if len(prior) else False
    occ2 = bool((prior.occlusion == 2).any()) if len(prior) else False
    if e3_type == "border" and (edge_px <= edge_max or oov2):
        return "partial-entry"
    if occ2 or (e3_type == "interior" and oov2):
        return "visibility-transition"
    if len(prior) == 0:
        return "annotation-late"
    return "true-birth"


def audit_event(gt_box, dets, b, prior, e3_type, W, H, seq_first=1):
    """Một sự kiện: nhãn, số khung khớp, onset_alt."""
    x, y, w, h = (float(v) for v in gt_box)
    edge = min(x, y, W - (x + w), H - (y + h))
    censored = b - K_PRE < seq_first
    m = det_matches(gt_box, dets, b, seq_first)
    hit = [f for f, v in m.items() if v >= IOU_MIN]
    n = len(hit)
    return dict(label=classify(n, censored, e3_type, edge, prior), label_edge5=classify(n, censored, e3_type, edge, prior, EDGE_PX_SENS),
                n_match=None if censored else n, edge_px=float(edge),
                n_prior_gt=int(len(prior)), iou_max=[m[f] for f in sorted(m)], onset_alt=int(min(hit)) if hit else int(b),
                shift=int(b - min(hit)) if hit else 0)


def placebo(gt_box, dets, b, W, H, rng):
    w, h = float(gt_box[2]), float(gt_box[3])
    box = (rng.uniform(0, max(W - w, 1)), rng.uniform(0, max(H - h, 1)), w, h)
    return det_matches(box, dets, b - 2, b - 3, k_pre=1).get(b - 3, 0.0) >= IOU_MIN


def audit(e3, rows, dets_by_seq, meta, split, seed=42):
    ev = e3[(e3.split == split) & (e3.vis_def == "main") & (~e3.never_visible)]
    gt = rows[rows.split == split].set_index(["seq", "tid", "frame"]).sort_index()
    rng = np.random.default_rng(seed)
    out = []
    for e in ev.itertuples(index=False):
        b, W, H = int(e.start), meta[e.seq]["W"], meta[e.seq]["H"]
        tr = gt.loc[(e.seq, e.track_id)]
        box = tr.loc[b, ["x", "y", "w", "h"]].values
        prior = tr[(tr.index >= b - K_PRE) & (tr.index < b)][["occlusion", "out_of_view"]]
        r = audit_event(box, dets_by_seq[e.seq], b, prior, e.e3_type, W, H, int(e.seq_first))
        r["placebo_hit"] = None if r["label"] == "censored-start" else bool(placebo(box, dets_by_seq[e.seq], b, W, H, rng))
        out.append(dict(seq=e.seq, track_id=int(e.track_id), start=b, birth_frame=int(e.birth_frame), D=int(e.D),
                        e3_type=e.e3_type, M_at_birth=float(e.M_at_birth), **r))
    return pd.DataFrame(out)


def class_table(a, col="label"):
    t = {}
    for typ in ("border", "interior", "all"):
        g = a if typ == "all" else a[a.e3_type == typ]
        t[typ] = {lb: int((g[col] == lb).sum()) for lb in LABELS}
        t[typ]["n"] = int(len(g))
        t[typ]["share_annotation_late"] = float((g[col] == "annotation-late").mean()) if len(g) else None
    return t


VISUAL_SEEN = {"partial-entry", "visibility-transition", "annotation-late"}


def compare_visual(au, vis, col="visual_v2"):
    """Đồng thuận nhãn tự động vs nhận xét crop (model-assisted). Báo cả đồng thuận đúng nhãn (5 lớp) và đồng thuận
    "đã thấy xe trước onset" (nhị phân)."""
    m = vis[vis[col] != "na"].merge(au[["seq", "track_id", "e3_type", "label", "n_match"]], on=["seq", "track_id"])
    ag = m[col] == m.label
    ag_bin = m[col].isin(VISUAL_SEEN) == m.label.isin(VISUAL_SEEN)
    return dict(kind="model-assisted (Claude xem crop start−2 … start+1), không phải nhãn người", n_crops=int(len(vis)),
                n_evaluable=int(len(m)), n_agree=int(ag.sum()), agreement=float(ag.mean()) if len(m) else None,
                n_agree_binary=int(ag_bin.sum()), agreement_binary=float(ag_bin.mean()) if len(m) else None,
                disagreements=m[~ag][["audit_id", "seq", "track_id", "e3_type", "label", "n_match", col]].rename(columns={col: "visual"})
                .to_dict("records"), note="bất đồng không đổi nhãn tự động")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--allow-test", action="store_true", help="CHỈ sau freeze PREREG (L8)")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("L8: TEST cấm trước freeze (dùng --allow-test sau khi đóng băng PREREG)")
    e3 = pd.read_parquet(P0 / "events_E3_uavdt.parquet")
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "occlusion", "out_of_view"])
    meta = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    seqs = sorted(e3[e3.split == a.split].seq.unique())
    dets = {s: pd.read_parquet(DUMPS_DIR / a.dump / f"uavdt_{s}.parquet", columns=["frame", "x", "y", "w", "h", "score"]) for s in seqs}
    au = audit(e3, rows, dets, meta, a.split)
    tm = class_table(au)
    pl = au.placebo_hit.dropna().astype(bool)
    out = dict(split=a.split, dump=a.dump, policy=POLICY, onset="start (khung VISIBLE đầu tiên)",
               rule=dict(k_pre=K_PRE, k_late=K_LATE, iou_min=IOU_MIN, score_min=SCORE_MIN, edge_px=EDGE_PX,
                         ref_box="GT box of the track at onset frame b", classes="all detector classes (as detector recall)",
                         consequence="none — descriptive only (PREREG §5, P2G-Q5)"),
               n_events=int(len(au)), table_main=class_table(au),
               table_edge5=class_table(au, "label_edge5"), edge5_note=f"độ nhạy: partial-entry với cách mép ≤ {EDGE_PX_SENS} px (chỉ mô tả)",
               placebo=dict(n=int(len(pl)), hit_rate=float(pl.mean()) if len(pl) else None, frame="b−3", seed=42),
               onset_alt=dict(n_shifted=int((au["shift"] > 0).sum()), shift_mean=float(au["shift"].mean()),
                              shift_by_label={lb: float(au[au.label == lb]["shift"].mean()) for lb in LABELS if (au.label == lb).any()}),
               events=au.to_dict("records"))
    out["interior_share_annotation_late"] = tm["interior"]["share_annotation_late"]
    out["limitation_flag_gt_20pct_interior"] = bool((tm["interior"]["share_annotation_late"] or 0) > 0.20)
    vp = P0 / f"audit_short_visual_{a.split}.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        if "visual_v2" in v:
            out["visual_check"] = compare_visual(au, v)
            print("visual_check", json.dumps({k: x for k, x in out["visual_check"].items() if k != "disagreements"}, ensure_ascii=False))
    p = P0 / f"audit_prebirth_{a.split}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("table_main", json.dumps(out["table_main"], ensure_ascii=False))
    print("placebo", out["placebo"], "onset_alt", out["onset_alt"])
    print("wrote", p)


if __name__ == "__main__":
    main()
