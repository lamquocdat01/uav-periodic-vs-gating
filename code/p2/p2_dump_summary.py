"""P2E-C: tóm tắt dump detector thật trong paths.DUMPS_DIR/<name>/ → results/p2/dump_queue_status.json (+ in 1 dòng/dump).

Mỗi dump: số chuỗi đã xong / số chuỗi của split, số khung, t_ms/khung (median meta.json), số box TB/khung (score ≥ 0,05 = mọi
box đã lưu, và score ≥ 0,25). Chỉ đọc; chạy lại được bất cứ lúc nào (kể cả khi dump đang chạy — chuỗi đang ghi chưa tính).
Chạy: python code/p2/p2_dump_summary.py [--names yolo26s_1024 yolo26n_1024 ...] [--dataset uavdt] [--split train]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "replay"))
from p2_dump_schema import DUMPS  # noqa: E402
from p2_frames import seqs_by_split  # noqa: E402

OUT = ROOT / "results" / "p2" / "dump_queue_status.json"


def summarize(name, dataset="uavdt", split="train"):
    d = DUMPS / name
    seqs = sorted(s for s, sp in seqs_by_split(dataset).items() if sp == split)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    nf = meta.get("n_frames", {})
    done = [s for s in seqs if (d / f"{dataset}_{s}.parquet").exists() and s in nf]
    frames = sum(nf[s] for s in done)
    boxes = boxes25 = 0
    for s in done:
        sc = pd.read_parquet(d / f"{dataset}_{s}.parquet", columns=["score"]).score
        boxes += len(sc)
        boxes25 += int((sc >= 0.25).sum())
    return dict(name=name, dataset=dataset, split=split, seqs_done=len(done), seqs_total=len(seqs), frames=frames,
                t_ms_median=meta.get("t_ms_median"), boxes_per_frame=boxes / frames if frames else None,
                boxes_per_frame_conf25=boxes25 / frames if frames else None, device=meta.get("params", {}).get("device"),
                model_dir=meta.get("params", {}).get("model_dir"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=None, help="mặc định: mọi thư mục dump không phải sim_*")
    ap.add_argument("--dataset", default="uavdt")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    names = a.names
    if names is None:
        names = sorted(p.name for p in DUMPS.iterdir() if p.is_dir() and not p.name.startswith("sim_")) if DUMPS.exists() else []
    rows = [summarize(n, a.dataset, a.split) for n in names]
    for r in rows:
        bpf = f"{r['boxes_per_frame']:.1f}" if r["boxes_per_frame"] is not None else "-"
        tms = f"{r['t_ms_median']:.1f}" if r["t_ms_median"] is not None else "-"
        print(f"{r['name']}: {r['seqs_done']}/{r['seqs_total']} chuỗi, {r['frames']} khung, {tms} ms/khung, {bpf} box/khung")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(updated=dt.datetime.now().isoformat(timespec="seconds"), dumps=rows), indent=1,
                              ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
