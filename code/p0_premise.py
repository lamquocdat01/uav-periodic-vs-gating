"""P0 bước 7: truy nguồn premise — CHỈ ĐỌC file của đề tài khác (ngoại lệ được phép của luật L1),
không import / không chạy code của chúng. Trích con số bằng regex/json vào p0_stats.json["premise"].
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "results" / "p0_stats.json"
PHD = Path(r"<workspace>")
SRC = {
    "deep2_tex": PHD / "Deep 2 Submission" / "deep2_manuscript.tex",
    "deep3_tex": PHD / "03.Final Submission" / "09.05 Certifiable worst-case" / "Revision_IoTJ_Sep.2026" / "manuscript" / "Deep3_manuscript.tex",
    "n3_json": PHD / "content-aware-frameskip-evaluation" / "results" / "n3_rareevent.json",
    "n1_json": PHD / "content-aware-frameskip-evaluation" / "results" / "n1_crossdata.json",
    "crc_summary_py": PHD / "publish" / "online-crc-skip-decisions" / "code" / "make_summary.py",
    "methodology_tex": PHD / "content-aware-frameskip-evaluation" / "paper" / "methodology_paper.tex",
}


def find_line(path, pat):
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = re.search(pat, line)
        if m:
            return i, m
    return None, None


def main():
    pr = {"sources": {k: str(v) for k, v in SRC.items()}}
    ln, m = find_line(SRC["deep2_tex"], r"nightVideos\s*&\s*(\d+)\s*&\s*([0-9.]+)")
    pr["deep2_nightVideos"] = dict(line=ln, n_events=int(m.group(1)), share_D_lt_8=float(m.group(2)),
                                   implied_n_lt_8=round(int(m.group(1)) * float(m.group(2))))
    ln2, _ = find_line(SRC["deep2_tex"], r"85\.6\\%\$? of its events are shorter than eight frames")
    pr["deep2_nightVideos"]["text_line"] = ln2
    n3 = json.loads(SRC["n3_json"].read_text(encoding="utf-8"))["mechanism"]["sparse_duration_law"]
    pr["sparse_regime"] = dict(n_events=int(n3["n_events"]), median=float(n3["median"]), frac_single=float(n3["frac_single"]),
                               cv=float(n3["cv"]), key="mechanism.sparse_duration_law")
    n1 = json.loads(SRC["n1_json"].read_text(encoding="utf-8"))["duration_stats"]
    pr["pooled_corpus"] = dict(n_events=int(sum(v["n_events"] for v in n1.values())),
                               by_dataset={k: dict(n_events=v["n_events"], median=v["median"]) for k, v in n1.items()})
    ln3, m3 = find_line(SRC["crc_summary_py"], r"median duration (\d+), ([0-9.]+)% of length 1")
    pr["pooled_corpus"].update(frac_single=float(m3.group(2)) / 100, median=float(m3.group(1)), frac_single_line=ln3)
    pr["pooled_corpus"]["frac_multi_D_ge_2"] = 1 - pr["pooled_corpus"]["frac_single"]
    ln4, _ = find_line(SRC["deep3_tex"], r"dominated by single-frame events")
    ln5, m5 = find_line(SRC["deep3_tex"], r"F_D\(4\)=([0-9.]+)")
    pr["deep3"] = dict(dominated_line=ln4, F_D4=float(m5.group(1)), F_D4_line=ln5)
    ln6, _ = find_line(SRC["methodology_tex"], r"dominated by single-frame events")
    pr["methodology_dominated_line"] = ln6
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}
    stats["premise"] = pr
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(pr, indent=1))


if __name__ == "__main__":
    main()
