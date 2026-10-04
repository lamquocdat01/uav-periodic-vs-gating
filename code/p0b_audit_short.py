"""P0b-A6: kiểm chứng đuôi ngắn E3 (D<8) bằng mắt.

1) Chọn mẫu (chạy được ngay, chỉ cần annotation): sự kiện E3 (VEHICLE, VISIBLE chính, CỬA SỔ SINH W=120)
   có D<8; phân tầng 20 border + 20 interior, seed=42. Nếu một tầng có ít hơn 20 sự kiện → lấy hết và ghi rõ.
   → results/p0/audit_short_selection.csv + AUDIT_SHORT_E3.md (bảng để anh Đạt gán nhãn).
2) Xuất ảnh (khi đã có frames trong paths.UAVDT_FRAMES/<seq>/img%06d.jpg; crop → paths.AUDIT_SHORT_DIR):
   mỗi sự kiện: khung start−2 … start+D+1 → crop quanh bbox (đệm 3× cạnh lớn, tối thiểu 96 px) có vẽ bbox,
   + 1 ảnh toàn khung thu nhỏ ở khung start. Lưu results/p0/audit_short/<seq>_<tid>/ (CHỈ local, git-ignored).
Nhãn: real_pass (xe thật lướt qua) / clip (cắt mép, bbox dính biên) / annot_error (lỗi annotation) / unsure.
Chạy: python code/p0b_audit_short.py [select|export|all] [train|test|all]   (mặc định all; export tự bỏ qua nếu chưa có frames;
      split mặc định train — P2E: TEST cấm, sự kiện chuỗi test không xuất)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from p0_stats import birth_window, censor_flags  # noqa: E402
from paths import AUDIT_SHORT_DIR, UAVDT_FRAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results" / "p0"
IMG = UAVDT_FRAMES
OUTDIR = AUDIT_SHORT_DIR
SEL = P0 / "audit_short_selection.csv"
MD = ROOT / "AUDIT_SHORT_E3.md"
N_PER, SEED, D_MAX = 20, 42, 8


def select():
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    e3 = pd.read_parquet(P0 / "events_E3_uavdt.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main")]
    e3 = censor_flags(e3, fr.groupby("seq").frame.max(), fr.groupby("seq").frame.min())
    pool = birth_window(e3)
    pool = pool[pool.D < D_MAX].sort_values(["seq", "track_id"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    parts, avail = [], {}
    for t in ("border", "interior"):
        g = pool[pool.e3_type == t]
        avail[t] = len(g)
        k = min(N_PER, len(g))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    sel = pd.concat(parts, ignore_index=True)
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet")
    info = []
    for _, e in sel.iterrows():
        d = rows[(rows.seq == e.seq) & (rows.tid == e.track_id) & (rows.frame >= e.start) & (rows.frame < e.start + e.D)]
        f0 = d.iloc[0]
        info.append(dict(x0=float(f0.x), y0=float(f0.y), w0=float(f0.w), h0=float(f0.h), sqrt_area=float(d.sqrt_area.median())))
    sel = pd.concat([sel, pd.DataFrame(info)], axis=1)
    sel["audit_id"] = [f"A{i + 1:02d}" for i in range(len(sel))]
    keep = ["audit_id", "seq", "track_id", "e3_type", "birth_frame", "start", "D", "category", "x0", "y0", "w0", "h0", "sqrt_area",
            "M_at_birth", "altitude"]
    sel[keep].to_csv(SEL, index=False)
    L = ["# AUDIT_SHORT_E3 — kiểm chứng đuôi ngắn E3 (D<8), UAVDT", "",
         f"*Sinh bởi `code/p0b_audit_short.py` (seed={SEED}). Mẫu: E3 VEHICLE, VISIBLE chính, cửa sổ sinh W=120, D<{D_MAX}. "
         f"Có sẵn: border {avail['border']}, interior {avail['interior']} → chọn {min(N_PER, avail['border'])} + {min(N_PER, avail['interior'])}"
         + (" (interior không đủ 20 → lấy hết)." if avail["interior"] < N_PER else ".") + "*", "",
         "Ảnh: `results/p0/audit_short/<seq>_<track>/` (chỉ local, không commit). **Chưa có frames UAVDT → ảnh chưa xuất**; chạy lại "
         "`python code/p0b_audit_short.py export` sau khi giải nén UAV-benchmark-M.", "",
         "Nhãn (điền cột *Nhãn*): `real_pass` = xe thật lướt qua góc/mép; `clip` = cắt mép (bbox dính biên, xe chỉ lộ một phần); "
         "`annot_error` = lỗi annotation (ID đổi, box sai, che khuất ghi nhầm); `unsure`.", "",
         "| ID | seq | track | loại | khung sinh | start | D | loại xe | bbox đầu (x,y,w,h) | √area | M lúc sinh | độ cao | Nhãn | Ghi chú |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sel.iterrows():
        L.append(f"| {r.audit_id} | {r.seq} | {r.track_id} | {r.e3_type} | {r.birth_frame} | {r.start} | {r.D} | {r.category} | "
                 f"{r.x0:.0f},{r.y0:.0f},{r.w0:.0f},{r.h0:.0f} | {r.sqrt_area:.1f} | {r.M_at_birth} | {r.altitude} |  |  |")
    L += ["", "## Tổng hợp (điền sau khi gán nhãn)", "", "| Loại | n | real_pass | clip | annot_error | unsure |", "|---|---|---|---|---|---|",
          "| border | | | | | |", "| interior | | | | | |"]
    MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("selected", len(sel), avail)
    return sel


def export(sel=None, split="train"):
    """split: chỉ xuất sự kiện thuộc chuỗi của split đó (P2E: TEST cấm → mặc định train; 'all' = mọi chuỗi)."""
    import cv2
    if sel is None:
        sel = pd.read_csv(SEL)
    if split != "all":
        meta = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
        skip = sel[sel.seq.map(lambda s: meta[s]["split"]) != split]
        sel = sel.drop(skip.index)
        print(f"split={split}: xuất {len(sel)} sự kiện, bỏ {len(skip)} ({', '.join(skip.audit_id)})")
    if not IMG.exists() or not any(IMG.iterdir()):
        print(f"chưa có frames UAVDT ({IMG}) → bỏ qua export")
        return 0
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet")
    n = 0
    for _, e in sel.iterrows():
        d = rows[(rows.seq == e.seq) & (rows.tid == e.track_id)].set_index("frame")
        od = OUTDIR / f"{e.seq}_{e.track_id}"
        od.mkdir(parents=True, exist_ok=True)
        side = max(96, int(3 * max(e.w0, e.h0)))
        cx, cy = e.x0 + e.w0 / 2, e.y0 + e.h0 / 2
        for f in range(int(e.start) - 2, int(e.start + e.D) + 2):
            p = IMG / e.seq / f"img{f:06d}.jpg"
            im = cv2.imread(str(p))
            if im is None:
                continue
            H, W = im.shape[:2]
            if f in d.index:
                b = d.loc[f]
                col = (0, 255, 0) if e.start <= f < e.start + e.D else (0, 165, 255)
                cv2.rectangle(im, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), col, 1)
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            crop = im[y0:min(H, y0 + side), x0:min(W, x0 + side)]
            cv2.imwrite(str(od / f"f{f:06d}_crop.jpg"), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            if f == int(e.start):
                cv2.imwrite(str(od / f"f{f:06d}_full.jpg"), cv2.resize(im, (W // 2, H // 2)))
            n += 1
    print("exported", n, "images →", OUTDIR)
    return n


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    s = select() if what in ("select", "all") else None
    if what in ("export", "all"):
        export(s, sys.argv[2] if len(sys.argv) > 2 else "train")
