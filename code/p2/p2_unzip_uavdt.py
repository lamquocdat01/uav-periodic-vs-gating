"""P2D-D1: giải nén UAV-benchmark-M.zip -> UAVDT_FRAMES và đối chiếu.

Kiểm: sha256 zip; mỗi entry file <-> file giải nén (tồn tại, cùng số byte, cùng CRC32 = byte-identical theo zip);
số ảnh mỗi chuỗi vs n_frames P0 (GT); kích thước ảnh thật (đọc header mọi ảnh) vs W/H tạm của P0.
Không xoá zip (xoá bằng tay, -WhatIf trước, khi verify PASS).
Usage: python code/p2/p2_unzip_uavdt.py [--no-extract] [--json-out results/p2/uavdt_frames_verify.json]
"""
import argparse
import collections
import hashlib
import json
import sys
import time
import zipfile
import zlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paths import UAVDT_FRAMES, UAVDT_ZIPS  # noqa: E402

ZIP = UAVDT_ZIPS / "UAV-benchmark-M.zip"
DEST = UAVDT_FRAMES.parent  # zip có thư mục gốc UAV-benchmark-M/


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def crc32(p, bs=1 << 22):
    c = 0
    with open(p, "rb") as f:
        while b := f.read(bs):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--json-out", default=str(ROOT / "results" / "p2" / "uavdt_frames_verify.json"))
    a = ap.parse_args()
    t0 = time.time()
    out = {"zip": str(ZIP), "zip_bytes": ZIP.stat().st_size}
    print("sha256 ...", flush=True)
    out["zip_sha256"] = sha256(ZIP)
    print(out["zip_sha256"], f"{time.time() - t0:.0f} s", flush=True)

    zf = zipfile.ZipFile(ZIP)
    files = [i for i in zf.infolist() if not i.is_dir()]
    out["entries_total"] = len(zf.infolist())
    out["entries_files"] = len(files)
    if not a.no_extract:
        print(f"extract {len(files)} files -> {DEST}", flush=True)
        for k, i in enumerate(files):
            tgt = DEST / i.filename
            if tgt.exists() and tgt.stat().st_size == i.file_size:
                continue  # resume; CRC kiểm ở bước sau
            zf.extract(i, DEST)
            if k % 5000 == 0:
                print(f"  {k}/{len(files)} {time.time() - t0:.0f} s", flush=True)

    print("verify entries ...", flush=True)
    bad = []
    for i in files:
        tgt = DEST / i.filename
        if not tgt.exists():
            bad.append((i.filename, "missing"))
        elif tgt.stat().st_size != i.file_size:
            bad.append((i.filename, "size"))
        elif crc32(tgt) != i.CRC:
            bad.append((i.filename, "crc"))
    out["entries_ok"] = len(files) - len(bad)
    out["entries_bad"] = bad[:50]
    on_disk = sum(1 for p in UAVDT_FRAMES.rglob("*") if p.is_file())
    out["files_on_disk"] = on_disk

    print("per-seq counts + image sizes ...", flush=True)
    meta = json.loads((ROOT / "results" / "p0" / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    seqs = {}
    for d in sorted(p for p in UAVDT_FRAMES.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        sizes = collections.Counter()
        for p in imgs:
            with Image.open(p) as im:
                sizes[im.size] += 1
        m = meta.get(d.name, {})
        idx = [int(p.stem.replace("img", "")) for p in imgs]
        seqs[d.name] = {
            "split": m.get("split"), "n_img": len(imgs), "n_frames_gt": m.get("n_frames"),
            "img_first": min(idx), "img_last": max(idx), "contiguous": idx == list(range(min(idx), max(idx) + 1)),
            "sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
            "p0_WH": [m.get("W"), m.get("H")],
        }
    out["seqs"] = seqs
    out["n_seqs"] = len(seqs)
    out["seqs_missing_vs_p0"] = sorted(set(meta) - set(seqs))
    out["seqs_extra_vs_p0"] = sorted(set(seqs) - set(meta))
    out["n_img_total"] = sum(s["n_img"] for s in seqs.values())
    out["n_frames_gt_total"] = sum(s["n_frames_gt"] or 0 for s in seqs.values())
    out["seqs_img_ne_gt"] = {k: [s["n_img"], s["n_frames_gt"]] for k, s in seqs.items() if s["n_img"] != s["n_frames_gt"]}
    out["seqs_noncontiguous"] = [k for k, s in seqs.items() if not s["contiguous"]]
    allsz = collections.Counter()
    for s in seqs.values():
        for k, n in s["sizes"].items():
            allsz[k] += n
    out["image_sizes_total"] = dict(allsz)
    for sp in ("train", "test"):
        ss = [s for s in seqs.values() if s["split"] == sp]
        out[f"{sp}_seqs"], out[f"{sp}_img"] = len(ss), sum(s["n_img"] for s in ss)
    out["verify_pass"] = (not bad and on_disk == len(files) and not out["seqs_missing_vs_p0"])
    out["elapsed_s"] = round(time.time() - t0)
    Path(a.json_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: v for k, v in out.items() if k != "seqs"}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
