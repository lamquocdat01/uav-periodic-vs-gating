"""P2A-A1: định dạng dump detector chuẩn + kiểm hợp lệ.

paths.DUMPS_DIR/<detector>_<imgsz>/<dataset>_<seq>.parquet — 1 dòng/box:
  frame (int ≥ 1), x, y, w, h (float, px, góc trên-trái), score (float ∈ [0,1]), cls (int)
Khung không có box nào vẫn hợp lệ (không có dòng). Kèm meta.json trong thư mục dump:
  detector, imgsz, weights_sha256 (hoặc null nếu giả lập), simulated (bool), t_ms_median (hoặc null),
  n_frames {seq: số khung}, created, params (dict tuỳ ý)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from paths import DUMPS_DIR as DUMPS  # noqa: E402
COLS = {"frame": "int", "x": "float", "y": "float", "w": "float", "h": "float", "score": "float", "cls": "int"}
META_KEYS = ["detector", "imgsz", "weights_sha256", "simulated", "t_ms_median", "n_frames", "created", "params"]


class DumpError(ValueError):
    pass


def validate(df, n_frames=None):
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise DumpError(f"thiếu cột: {missing}")
    for c, kind in COLS.items():
        ok = pd.api.types.is_integer_dtype(df[c]) if kind == "int" else pd.api.types.is_numeric_dtype(df[c])
        if not ok:
            raise DumpError(f"cột {c} sai kiểu ({df[c].dtype}), cần {kind}")
    if len(df):
        if df[["x", "y", "w", "h", "score"]].isna().any().any():
            raise DumpError("có NaN")
        if (df.frame < 1).any() or (n_frames is not None and (df.frame > n_frames).any()):
            raise DumpError("frame ngoài [1, n_frames]")
        if ((df.w <= 0) | (df.h <= 0)).any():
            raise DumpError("w/h ≤ 0")
        if ((df.score < 0) | (df.score > 1)).any():
            raise DumpError("score ngoài [0,1]")
    return True


def validate_meta(meta):
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise DumpError(f"meta thiếu khoá: {missing}")
    return True


def write_dump(dirpath, dataset, seq, df, n_frames):
    df = df[list(COLS)].copy()
    df["frame"] = df.frame.astype(np.int32)
    df["cls"] = df.cls.astype(np.int16)
    for c in ("x", "y", "w", "h", "score"):
        df[c] = df[c].astype(np.float32)
    validate(df, n_frames)
    dirpath.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dirpath / f"{dataset}_{seq}.parquet", index=False)


def write_meta(dirpath, meta):
    validate_meta(meta)
    (dirpath / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def read_dump(dirpath, dataset, seq):
    df = pd.read_parquet(dirpath / f"{dataset}_{seq}.parquet")
    validate(df)
    return df


def validate_dir(dirpath):
    meta = json.loads((dirpath / "meta.json").read_text(encoding="utf-8"))
    validate_meta(meta)
    n = 0
    for p in sorted(dirpath.glob("*.parquet")):
        seq = p.stem.split("_", 1)[1]
        validate(pd.read_parquet(p), meta["n_frames"].get(seq))
        n += 1
    return n


if __name__ == "__main__":
    import sys
    for d in (sys.argv[1:] or [str(p) for p in DUMPS.iterdir() if p.is_dir()] if DUMPS.exists() else []):
        print(d, "OK", validate_dir(Path(d)), "files")
