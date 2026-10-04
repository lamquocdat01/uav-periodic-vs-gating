"""Tiện ích chung P2: đường dẫn khung ảnh theo dataset/chuỗi + danh sách chuỗi theo split (từ kết quả P0)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paths import UAVDT_FRAMES, visdrone_split  # noqa: E402

P0 = ROOT / "results" / "p0"


def frame_paths(dataset, seq, split=None):
    """list ảnh theo thứ tự khung (khung 1 = phần tử 0). Rỗng nếu chưa có frames."""
    if dataset == "uavdt":
        d = UAVDT_FRAMES / seq
    else:
        d = visdrone_split(split) / "sequences" / seq
    return sorted(p for p in d.glob("*") if p.suffix.lower() in (".jpg", ".png")) if d.exists() else []


def seqs_by_split(dataset):
    """{seq: split} từ parse_<ds>.json (P0)."""
    p = P0 / f"parse_{dataset}.json"
    if not p.exists():
        return {}
    meta = json.loads(p.read_text(encoding="utf-8")).get("seq_meta", {})
    return {s: m["split"] for s, m in meta.items()}
