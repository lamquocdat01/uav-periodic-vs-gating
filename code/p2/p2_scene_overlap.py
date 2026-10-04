"""P2A-B5: kiểm trùng cảnh giữa VisDrone2019-DET train (dùng train YOLO) và các chuỗi TEST
(VisDrone-VID val + test-dev, UAVDT test).

Hai phép kiểm:
  (1) tên: tiền tố chuỗi của ảnh DET (vd "0000002_00005_d_0000014" → "0000002") so với tên chuỗi VID/UAVDT;
  (2) perceptual hash (pHash: DCT 32×32 xám → 8×8 hệ số tần thấp bỏ DC, so với trung vị → 64 bit) của khung đầu/giữa/cuối
      mỗi chuỗi TEST vs MỌI ảnh DET-train; gần trùng nếu khoảng cách Hamming ≤ 6.
Xuất results/p2/scene_overlap.json: % chuỗi TEST có ≥1 ảnh DET-train gần trùng, danh sách cặp.
Chạy: python code/p2/p2_scene_overlap.py --det-train <thư mục ảnh VisDrone2019-DET-train/images> [--max-hamming 6]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from p2_frames import frame_paths, seqs_by_split  # noqa: E402

OUT = ROOT / "results" / "p2" / "scene_overlap.json"
MAX_HAMMING = 6


def phash(img_or_path):
    img = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE) if not isinstance(img_or_path, np.ndarray) else img_or_path
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(s)[:8, :8].flatten()[1:]  # bỏ DC
    bits = d > np.median(d)
    return np.packbits(np.r_[bits, False])  # 64 bit (63 + 1 đệm)


def hamming_matrix(A, B):
    """A: (n, 8) uint8, B: (m, 8) → (n, m) khoảng cách Hamming."""
    x = np.bitwise_xor(A[:, None, :], B[None, :, :])
    return np.unpackbits(x, axis=2).sum(axis=2)


def test_sequences():
    """(dataset, seq, split) của các chuỗi TEST có frames."""
    out = []
    for s, sp in seqs_by_split("uavdt").items():
        if sp == "test":
            out.append(("uavdt", s, sp))
    for s, sp in seqs_by_split("visdrone").items():
        if sp in ("val", "test-dev"):
            out.append(("visdrone", s, sp))
    return out


def run(det_dir, test_seqs, max_h=MAX_HAMMING):
    det_imgs = sorted(p for p in Path(det_dir).glob("*") if p.suffix.lower() in (".jpg", ".png"))
    H_det = np.stack([phash(p) for p in det_imgs]) if det_imgs else np.zeros((0, 8), np.uint8)
    det_prefix = {p.stem.split("_")[0] for p in det_imgs}
    rows = []
    for ds, seq, sp in test_seqs:
        fp = frame_paths(ds, seq, sp) if isinstance(seq, str) else seq
        if not fp:
            rows.append(dict(dataset=ds, seq=seq, split=sp, status="NO_FRAMES"))
            continue
        pick = [fp[0], fp[len(fp) // 2], fp[-1]]
        Hs = np.stack([phash(p) for p in pick])
        dm = hamming_matrix(Hs, H_det) if len(H_det) else np.full((3, 0), 64)
        near = [(pick[i].name, det_imgs[j].name, int(dm[i, j])) for i, j in zip(*np.nonzero(dm <= max_h))]
        seq_prefix = str(seq).split("_")[0]
        rows.append(dict(dataset=ds, seq=seq, split=sp, status="OK", name_match=bool(seq_prefix in det_prefix),
                         min_hamming=(int(dm.min()) if dm.size else None), n_near=len(near), near=near[:20]))
    ok = [r for r in rows if r["status"] == "OK"]
    return dict(det_dir=str(det_dir), n_det_images=len(det_imgs), max_hamming=max_h, n_test_seq=len(rows), n_with_frames=len(ok),
                share_seq_near_dup=(float(np.mean([r["n_near"] > 0 for r in ok])) if ok else None),
                share_seq_name_match=(float(np.mean([r["name_match"] for r in ok])) if ok else None), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-train", required=True)
    ap.add_argument("--max-hamming", type=int, default=MAX_HAMMING)
    a = ap.parse_args()
    out = run(a.det_train, test_sequences(), a.max_hamming)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
