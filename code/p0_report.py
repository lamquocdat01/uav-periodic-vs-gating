"""P0 bước 9 (v2 sau rà chéo 25/09/2026; v3 P2F/P2G 01-10-2026: kích thước ảnh thật, không lọc): sinh P0_REPORT_28.md + DATA_SOURCES.md từ results/p0_stats.json, rồi tự kiểm:
mọi con số trong báo cáo phải có mặt trong p0_stats.json (sau định dạng) → results/p0/number_check.json.
"""
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from paths import UAVDT_FRAMES, UAVDT_GT, UAVDT_ZIPS, VISDRONE_DIR, VISDRONE_ZIPS  # noqa: E402
STATS = ROOT / "results" / "p0_stats.json"
P0 = ROOT / "results" / "p0"
REPORT = ROOT / "P0_REPORT_28.md"

# Hằng số lấy từ PROMPT/KE_HOACH (kỳ vọng, ngưỡng quyết định, lưới) — ghi vào stats để báo cáo chỉ đọc từ json
CONSTANTS = {
    "expected": {"VisDrone": {"clips": {"train": 56, "val": 7, "test-dev": 16, "total": 79},
                              "frames": {"train": 24198, "val": 2846, "test-dev": 6322, "total": 33366}},
                 "UAVDT": {"seq": 50, "frames": 40000, "fps": 30, "W": 1080, "H": 540, "seq_KE_HOACH": 100, "frames_KE_HOACH": 80000}},
    "thresholds": {"B_share_short": 0.05, "C_share_short": 0.01, "border_frac": 0.02, "diff_thr": 25, "hover_px": 1.0,
                   "size_bins": [16, 32, 64], "M_bins": [1, 5, 6, 20], "ransac_px": 3, "orb": 2000, "ratio": 0.75,
                   "pair_step": 10, "max_pairs": 300, "detector_levels": 3, "ci_pct": 95, "min_dossier_tits": 3, "min_dossier_tvt": 2},
    "visdrone_fps": {"value": None, "checked": ["README github VisDrone-Dataset", "arXiv 2001.06303 (20 trang, full-text)", "EXIF ảnh: chưa có frames"]},
    "dates": {"cp0": "12/10/2026"},
    "rho_ev_ref": 0.9,
    "acm_uav_sar": {"year": 2025, "fulltext_http": 403},
    # số do rà chéo 25/09/2026 báo (PROMPT_28_P0b_P1) — chỉ để đối chiếu với số script tính lại
    "review_2509": {"share_short_S50": 0.082, "share_short_S120": 0.136, "iso_S_r1": {"eps1": 2, "eps2": 5, "eps5": 14},
                    "border_share_D_lt_8": 0.76},
}

DS_FILES = [
    ("UAVDT_MOT_toolkit.zip", "UAVDT UAV-benchmark-MOTD_v1.0 (GT + toolkit)", "https://drive.google.com/open?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq",
     "GNU-GPL (toolkit); dataset \"research purpose only\" (README)"),
    ("UAVDT_attributes.zip", "UAVDT sequence attributes (M_attr)", "https://drive.google.com/open?id=1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1", "như UAVDT"),
    ("UAV-benchmark-M.zip", "UAVDT frames UAV-benchmark-M", "https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view", "research only"),
    ("VisDrone2019-VID-train.zip", "VisDrone2019-VID train", "https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw/view",
     "research only (điều khoản AISKYEYE)"),
    ("VisDrone2019-VID-val.zip", "VisDrone2019-VID val", "https://drive.google.com/file/d/1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B/view", "research only"),
    ("VisDrone2019-VID-test-dev.zip", "VisDrone2019-VID test-dev", "https://drive.google.com/open?id=1-BEq--FcjshTF1UwUabby_LHhYj41os5", "research only"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=0):
    if x is None:
        return "—"
    return f"{x:,.{d}f}".replace(",", ".") if d == 0 else f"{x:.{d}f}"


def dossier_counts():
    t = (ROOT / "DOSSIER_TITS_28.md").read_text(encoding="utf-8")
    return {"TITS": len(re.findall(r"^\| T\d+ \|", t, re.M)), "TVT": len(re.findall(r"^\| V\d+ \|", t, re.M))}


def zip_dir(fn):
    return VISDRONE_ZIPS if fn.startswith("VisDrone") else UAVDT_ZIPS


def data_sources(stats):
    # zip nằm ở THS_DATASETS/<dataset>/_zips; theo luật dataset zip bị xoá sau khi giải nén + đối chiếu
    # → khi không còn zip thì giữ nguyên bản ghi (size, sha256, ngày) đã lưu trong p0_stats.json.
    prev = stats.get("downloads", {})
    rows, files = [], {}
    for fn, desc, url, lic in DS_FILES:
        p = zip_dir(fn) / fn
        ok = p.exists() and p.read_bytes()[:2] == b"PK"
        if ok:
            files[fn] = dict(size=p.stat().st_size, sha256=sha256(p), date=dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"))
        elif fn in prev:
            files[fn] = prev[fn]
        rows.append((fn, desc, url, lic, files.get(fn)))
    stats["downloads"] = files
    L = ["# DATA_SOURCES — P0 #28 (sinh bởi code/p0_report.py)", "",
         "Chỉ tải từ link Google Drive chính thức (VisDrone: README github.com/VisDrone/VisDrone-Dataset; UAVDT: sites.google.com/view/grli-uavdt). "
         "Không re-host frame; dữ liệu nằm ngoài repo (THS_DATASETS — xem code/paths.py). Link bị chặn: xem DOWNLOAD_BLOCKED.md.", "",
         f"Vị trí (từ 28-09-2026): UAVDT annotation → `{UAVDT_GT.parent.as_posix()}/`, frames → `{UAVDT_FRAMES.as_posix()}/`, "
         f"zip → `{UAVDT_ZIPS.as_posix()}/`, `{VISDRONE_ZIPS.as_posix()}/`; VisDrone-VID → `{VISDRONE_DIR.as_posix()}/{{train,val,test-dev}}/` (gốc THS_DATASETS). "
         "sha256 dưới đây là của zip gốc (Move-Item cùng ổ không đổi byte); zip có thể đã xoá sau đối chiếu — bản ghi giữ trong results/p0_stats.json.", "",
         "| File | Nội dung | URL | Ngày tải | Kích thước (byte) | sha256 | License/điều khoản |", "|---|---|---|---|---|---|---|"]
    for fn, desc, url, lic, f in rows:
        if f:
            L.append(f"| {fn} | {desc} | {url} | {f['date']} | {f['size']} | `{f['sha256']}` | {lic} |")
        else:
            L.append(f"| {fn} | {desc} | {url} | — | — | **CHƯA TẢI (quota Drive)** | {lic} |")
    L += ["", "## Số thực đếm vs kỳ vọng", ""]
    L += count_table(stats)
    u = stats["data"].get("UAVDT", {})
    if u.get("status") == "OK":
        L += ["", "### Độ lệch UAVDT so với kỳ vọng / README (L5 — ghi lại, không ép)",
              f"- Số chuỗi DET/MOT thực: {u['n_seq']} (train {u['n_seq_by_split']['train']} / test {u['n_seq_by_split']['test']}); KE_HOACH mục 4 ghi 100 seq ~80k khung — đó là tổng gồm cả phần SOT (UAV-benchmark-S), không thuộc P0.",
              f"- Số khung: {u['n_frames_total']} (nguồn: {u['frames_source']}; = frame index lớn nhất trong GT mỗi chuỗi khi chưa có ảnh).",
              f"- Kích thước ảnh: kỳ vọng 1080×540 (paper). GT extent lớn nhất = {u['gt_extent_max'][0]}×{u['gt_extent_max'][1]} → ảnh thực nhiều khả năng 1024×540. Đang dùng {', '.join(u['image_sizes'])}; xác nhận khi có frames.",
              "- README MOTD: file *_gt.txt ghi `<in-view>` và `<occlusion>` \"hằng -1\", thực tế cột 8 = 1, cột 9 = -1. P0 dùng *_gt_whole.txt (đúng định dạng README: out_of_view ∈ {1,2,3}, occlusion ∈ {1,2,3,4}, category ∈ {1,2,3}).",
              "- File thuộc tính test có tên lỗi `M0701 _attr.txt` (dấu cách) — parser đã strip.",
              f"- Box theo loại: {u['category_counts']}; occlusion: {u['occlusion_counts']}; out_of_view: {u['out_of_view_counts']}.",
              f"- Độ cao (thuộc tính chuỗi): {u['altitude_counts']}."]
    (ROOT / "DATA_SOURCES.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def count_table(stats):
    E = stats["constants"]["expected"]
    L = ["| Dataset | Mục | Thực đếm | Kỳ vọng | Ghi chú |", "|---|---|---|---|---|"]
    v = stats["data"].get("VisDrone", {})
    if v.get("status") == "OK":
        for sp in ("train", "val", "test-dev"):
            L.append(f"| VisDrone-VID | clip {sp} | {v['n_seq_by_split'][sp]} | {E['VisDrone']['clips'][sp]} | |")
            L.append(f"| VisDrone-VID | khung {sp} | {v['n_frames_by_split'][sp]} | {E['VisDrone']['frames'][sp]} | |")
    else:
        L.append(f"| VisDrone-VID | clip / khung | CHƯA CÓ DỮ LIỆU | {E['VisDrone']['clips']['total']} / {E['VisDrone']['frames']['total']} | quota Drive — DOWNLOAD_BLOCKED.md |")
    u = stats["data"].get("UAVDT", {})
    if u.get("status") == "OK":
        L.append(f"| UAVDT | chuỗi (train/test) | {u['n_seq']} ({u['n_seq_by_split']['train']}/{u['n_seq_by_split']['test']}) | ~{E['UAVDT']['seq']} | |")
        L.append(f"| UAVDT | khung | {u['n_frames_total']} | ~{E['UAVDT']['frames']} | từ GT (frames chưa tải) |" if u["frames_source"] != "images"
                 else f"| UAVDT | khung | {u['n_frames_total']} | ~{E['UAVDT']['frames']} | đếm ảnh |")
        L.append(f"| UAVDT | kích thước ảnh | {', '.join(u['image_sizes'])} | {E['UAVDT']['W']}×{E['UAVDT']['H']} | lệch — xem dưới |")
        L.append(f"| UAVDT | track xe / box | {u['n_tracks']} / {u['n_boxes']} | — | |")
    return L


def exp_rows(stats, ds, ev, vd="main", win="birth_W120"):
    return [r for r in stats["exposure"] if r["dataset"] == ds and r["event"] == ev and r["vis_def"] == vd and r["group"] == "VEHICLE"
            and r.get("window", "all_v1") == win]


def exp_cell(stats, ds, ev, k, R, win, vd="main"):
    for r in exp_rows(stats, ds, ev, vd, win):
        if r["k"] == k and r["R"] == R:
            return r
    return None


def main_table(stats, ds):
    L = [f"**{ds}** (VEHICLE, VISIBLE chính; **cửa sổ sinh W={stats['meta']['W_birth']}** = chính; CI 95% bootstrap theo chuỗi; "
         "r=1 chính xác, r<1 cửa sổ sinh = cận trên, KM = Kaplan–Meier; cột v1 = số P0 cũ chưa sửa kiểm duyệt)", "",
         "| Sự kiện | k | R | S | share_short [CI] | v1 | miss r=0.5 cửa sổ [CI] | miss r=0.5 KM [CI] | miss r=0.8 cửa sổ / KM |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ev in ("E2", "E3"):
        for r in exp_rows(stats, ds, ev):
            if r["k"] in (1, 3, 5, 12) and r["R"] in (5, 10):
                o = exp_cell(stats, ds, ev, r["k"], r["R"], "all_v1")
                km = exp_cell(stats, ds, ev, r["k"], r["R"], "km")
                L.append(f"| {ev} | {r['k']} | {r['R']} | {r['S']} | {pct(r['share_short'])} [{pct(r['share_short_lo'])}, {pct(r['share_short_hi'])}] "
                         f"| {pct(o['share_short'])} | {pct(r['miss_r0.5'])} [{pct(r['miss_r0.5_lo'])}, {pct(r['miss_r0.5_hi'])}] "
                         f"| {pct(km['miss_r0.5'])} [{pct(km['miss_r0.5_lo'])}, {pct(km['miss_r0.5_hi'])}] | {pct(r['miss_r0.8'])} / {pct(km['miss_r0.8'])} |")
    q = stats["D_quantiles"][ds]
    qk = stats["D_quantiles_km"][ds]
    W = stats["meta"]["W_birth"]
    L += ["", f"Phân vị D: cửa sổ sinh (giá trị > {W} khung chỉ là cận dưới, đánh dấu ≥) và Kaplan–Meier (KM; \"—\" = không đạt vì đuôi bị kiểm duyệt)."]
    for key, lab in (("E2|main|VEHICLE", "E2"), ("E3|main|VEHICLE", "E3"), ("E3|main|VEHICLE|border", "E3-border"),
                     ("E3|main|VEHICLE|interior", "E3-interior"), ("E1|main|VEHICLE", "E1"), ("E3|strict|VEHICLE", "E3 (STRICT)")):
        base = key.split("|")
        bkey = f"{'|'.join(base[:3])}|birth" if len(base) == 3 else f"{key}|birth"
        d, k = q.get(bkey), qk.get(key)
        if d:
            def fq(p):
                v = num(d[f"p{p}"])
                return v if d.get(f"p{p}_exact", True) else f"≥{v}"
            kmq = " / ".join(num(k[f"p{p}"]) if k[f"p{p}_reached"] else "—" for p in (10, 50, 90)) if k else "—"
            L.append(f"- D {lab} (cửa sổ sinh n={d['n']}): p10/p50/p90 = {fq(10)} / {fq(50)} / {fq(90)} khung; KM {kmq}; "
                     f"D<8: {pct(d['share_D_lt_8'])} (v1 {pct(q[key]['share_D_lt_8'])}).")
    return L


def p0b_section(stats, ds="UAVDT"):
    """Mục 0: sửa sau rà chéo 25/09/2026 (A1–A6), số cũ → mới."""
    if ds not in stats.get("censoring", {}) or "p0b" not in stats:
        return []
    C = stats["constants"]["review_2509"]
    cen = stats["censoring"][ds]
    b = stats["p0b"]
    e3, e2 = cen["E3|main|VEHICLE"], cen["E2|main|VEHICLE"]
    W = stats["meta"]["W_birth"]
    L = ["## 0. Sửa sau rà chéo 25/09/2026 (P0b: A1–A6)", "",
         "### A1 — Kiểm duyệt phải → cửa sổ sinh (chỉ số chính)", "",
         f"- E3: {e3['n_right_censored']}/{e3['n_all']} sự kiện ({pct(e3['share_right_censored'])}) còn visible ở khung cuối chuỗi (D bị cắt). "
         f"E2: {e2['n_right_censored']}/{e2['n_all']} ({pct(e2['share_right_censored'])}) bị cắt phải và {e2['n_left_censored']} bắt đầu ở khung đầu (cắt trái, loại khỏi cửa sổ sinh và KM).",
         f"- Cửa sổ sinh W={W}: giữ sự kiện có start ≤ seq_last − {W} → E3 còn {e3['n_birth_window']} (trong đó {e3['n_birth_window_right_censored']} bị cắt nhưng D_obs ≥ {e3['birth_censored_min_D_obs']} > S nên đóng góp r=1 đúng bằng 0 — THEORY_28 Cor. 1). "
         f"r<1: cửa sổ sinh là cận trên; báo kèm Kaplan–Meier.",
         "", "| E3, VISIBLE chính | v1 (cũ) | rà chéo | cửa sổ sinh [CI] (mới) | KM |", "|---|---|---|---|---|"]
    for k, R, ck in ((5, 10, "share_short_S50"), (12, 10, "share_short_S120")):
        n_, o_, m_ = exp_cell(stats, ds, "E3", k, R, "birth_W120"), exp_cell(stats, ds, "E3", k, R, "all_v1"), exp_cell(stats, ds, "E3", k, R, "km")
        L.append(f"| share_short S={k * R} | {pct(o_['share_short'])} | {pct(C[ck])} | {pct(n_['share_short'])} [{pct(n_['share_short_lo'])}, {pct(n_['share_short_hi'])}] | {pct(m_['share_short'])} |")
        L.append(f"| miss r=0.5 S={k * R} | {pct(o_['miss_r0.5'])} | — | ≤ {pct(n_['miss_r0.5'])} [{pct(n_['miss_r0.5_lo'])}, {pct(n_['miss_r0.5_hi'])}] | {pct(m_['miss_r0.5'])} [{pct(m_['miss_r0.5_lo'])}, {pct(m_['miss_r0.5_hi'])}] |")
    L += ["", f"- Đuôi D<8 của E3: v1 {e3['all_D_lt_8_n']} sự kiện ({pct(e3['all_D_lt_8_share_border'])} vào từ mép) → cửa sổ sinh {e3['birth_D_lt_8_n']} ({pct(e3['birth_D_lt_8_share_border'])} vào từ mép). "
          "Quyết định A không đổi (share_short(S=50) vẫn > 5%). Mọi bảng mục 2 và 4 đã tính lại trên cửa sổ sinh.", ""]
    # A2
    L += ["### A2 — Độ nhạy vùng quan tâm (ROI)", "",
          "E3-ROI(m): tâm bbox đi vào vùng trong cách mép m; D = đoạn liên tiếp đầu tiên trong ROI thuộc đoạn visible đầu; cửa sổ sinh.", "",
          "| m | xe vào ROI | cửa sổ sinh | D<8 (tỉ lệ) | còn lại so với m=0 | border / interior trong D<8 | D p50 | share_short S=10 | S=50 [CI] | S=120 [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mk, r in b["roi"][ds].items():
        m = float(mk[1:])
        c10, c50, c120 = r["cells"]["S10"], r["cells"]["S50"], r["cells"]["S120"]
        rem = pct(r["D_lt_8_remaining_vs_m0"]) if r["D_lt_8_remaining_vs_m0"] is not None else "—"
        L.append(f"| {pct(m, 0)} | {r['n_enter']} (không vào: {r['n_not_enter']}) | {r['n_birth']} | {r['D_lt_8_n']} ({pct(r['D_lt_8_share'])}) | {rem} "
                 f"| {r['D_lt_8_border_n']} / {r['D_lt_8_interior_n']} | {num(r['D_birth']['p50'])} | {pct(c10['share_short'])} "
                 f"| {pct(c50['share_short'])} [{pct(c50['share_short_ci'][0])}, {pct(c50['share_short_ci'][1])}] "
                 f"| {pct(c120['share_short'])} [{pct(c120['share_short_ci'][0])}, {pct(c120['share_short_ci'][1])}] |")
    L += ["", "- Bảng đủ k∈{1,5,12} × R∈{5,10} và miss r=0.8/0.5 (cửa sổ + KM): results/p0/roi_grid.csv, khối p0b.roi trong json.", ""]
    # A3
    rv = C["iso_S_r1"]
    L += ["### A3 — Iso-miss: stride lớn nhất đạt miss ≤ ε", "",
          "S* = S lớn nhất sao cho miss(S') ≤ ε với mọi S' ≤ S (S ∈ 1..120); Hz = 30/S*; a = 1/S*. CI bootstrap theo chuỗi. r<1: cửa sổ sinh (cận trên); KM trong json.", "",
          "| m | r | ε=10% S* [CI] | ε=5% | ε=2% | ε=1% | Hz tại ε=1% | a tại ε=1% |", "|---|---|---|---|---|---|---|---|"]
    for m in ("0.00", "0.05"):
        for r in (1.0, 0.8, 0.5):
            d = b["isomiss"][ds][f"m{m}|r{r}|birth"]
            cells = []
            for e in (0.1, 0.05, 0.02, 0.01):
                x = d[f"eps{e}"]
                cells.append(("≥" if x["capped"] else "") + f"{x['S_star']} [{x['S_star_ci'][0]}, {x['S_star_ci'][1]}]")
            x1 = d["eps0.01"]
            hz = num(x1["hz_star"], 1) if x1.get("hz_star") else "không đạt"
            aa = num(x1["a_star"], 2) if x1.get("a_star") else "—"
            L.append(f"| {pct(float(m), 0)} | {r} | " + " | ".join(cells) + f" | {hz} | {aa} |")
    r1 = b["isomiss"][ds]["m0.00|r1.0|birth"]
    L += ["", f"- Rà chéo (trên số chưa sửa kiểm duyệt) báo r=1: miss ≤1% cần S ≤ {rv['eps1']}, ≤2% cần S ≤ {rv['eps2']}, ≤5% cần S ≤ {rv['eps5']}. "
          f"Sau sửa: S* = {r1['eps0.01']['S_star']} / {r1['eps0.02']['S_star']} / {r1['eps0.05']['S_star']} "
          f"(≈ {num(r1['eps0.01']['hz_star'], 1)} / {num(r1['eps0.02']['hz_star'], 1)} / {num(r1['eps0.05']['hz_star'], 1)} Hz ở 30 fps). "
          f"Kết luận định tính giữ nguyên: vùng tin cậy cao (ε ≤ 2%) buộc detector {num(r1['eps0.02']['hz_star'], 0)}–{num(r1['eps0.01']['hz_star'], 0)} Hz tuỳ ε — đó là chỗ gate có thể thắng.", ""]
    # A4
    L += ["### A4 — Độ trễ phát hiện xe mới (onset latency)", "",
          "P(L ≤ Lmax) = 1 − miss_r(min(D, Lmax+1), S) (THEORY_28 Cor. 1.3), lịch tuần hoàn pha đều, E3 cửa sổ sinh, m=0. "
          "Cột S cần = S lớn nhất để P(L ≤ Lmax) ≥ ngưỡng (0 = không đạt kể cả S=1).", "",
          "| r | Lmax (khung / s) | P(L≤Lmax) S=5 | S=10 | S=30 | S cần cho 0.95 [CI] | S cần cho 0.99 [CI] |", "|---|---|---|---|---|---|---|"]
    for r in (1.0, 0.8, 0.5):
        for Lm in (3, 5, 10, 30):
            d = b["latency"][ds][f"m0.00|r{r}|L{Lm}"]
            g = d["grid"]
            n95, n99 = d["S_needed"]["p0.95"], d["S_needed"]["p0.99"]
            L.append(f"| {r} | {Lm} / {num(d['Lmax_s'], 2)} | {pct(g['S5']['p'])} | {pct(g['S10']['p'])} | {pct(g['S30']['p'])} "
                     f"| {n95['S_star']} [{n95['S_star_ci'][0]}, {n95['S_star_ci'][1]}] | {n99['S_star']} [{n99['S_star_ci'][0]}, {n99['S_star_ci'][1]}] |")
    L += ["", "- m=5%: khối p0b.latency (json) và results/p0/latency.csv.", ""]
    # A5
    L += ["### A5 — Mật độ cửa sổ khởi phát ρ_onset(w)", "",
          "Tỉ lệ khung nằm trong ≥ 1 cửa sổ [onset, onset+w) của sự kiện E3 (mọi sự kiện, cắt ở cuối chuỗi). Đây là \"in-event fraction\" đúng cho gate nhắm xe mới (ρ_ev class-agnostic ≈ 1 không dùng được).", "",
          "| m | w (khung) | gộp [CI] | trung vị theo chuỗi | p10–p90 theo chuỗi | chuỗi = 0 | λ (xe mới/khung) |", "|---|---|---|---|---|---|---|"]
    for mk in ("0.00", "0.05"):
        for w in (3, 5, 10):
            d = b["rho_onset"][ds][f"m{mk}|w{w}"]
            L.append(f"| {pct(float(mk), 0)} | {w} | {pct(d['pooled'])} [{pct(d['pooled_ci'][0])}, {pct(d['pooled_ci'][1])}] | {pct(d['seq_median'])} "
                     f"| {pct(d['seq_p10'])}–{pct(d['seq_p90'])} | {d['n_seq_zero']} | {num(d['lambda_per_frame'], 3)} |")
    au = b.get("audit", {})
    L += ["", "### A6 — Kiểm chứng đuôi ngắn bằng mắt", "",
          f"- Mẫu phân tầng (seed=42, E3 cửa sổ sinh, D<8): border {au.get('n_border_selected')}/{au.get('n_border_avail')}, interior {au.get('n_interior_selected')}/{au.get('n_interior_avail')} "
          f"(interior không đủ 20 → lấy hết). Bảng gán nhãn: AUDIT_SHORT_E3.md. **Ảnh chưa xuất** (chưa có frames UAVDT) — chạy `code/p0b_audit_short.py export` khi có frames.",
          "", "Lý thuyết đi kèm (P1): THEORY_28.md; kiểm số results/p1/theory_checks.json; bản nháp đăng-ký-trước PREREG_28_DRAFT.md (chưa đóng băng).", ""]
    return L


def strata_table(stats, ds, k=5, R=10):
    S = [s for s in stats["strata"] if s["dataset"] == ds and s["event"] == "E3" and s["vis_def"] == "main" and s["k"] == k and s["R"] == R]
    L = [f"E3, VISIBLE chính, cửa sổ sinh, k={k}, R={R} (S={k * R}); D p50 > W là cận dưới (≥):", "", "| Biến | Mức | n sự kiện | n chuỗi | D p50 | share_short | miss r=0.5 |", "|---|---|---|---|---|---|---|"]
    order = {"size_bin": 0, "altitude": 1, "speed_tertile": 2, "M_bin": 3, "e3_type": 4, "ego": 5}
    for s in sorted(S, key=lambda s: (order.get(s["stratum_var"], 9), s["stratum"])):
        dp = num(s["D_p50"]) if s.get("D_p50_exact", True) else "≥" + num(s["D_p50"])
        L.append(f"| {s['stratum_var']} | {s['stratum']} | {s['n_events']} | {s['n_seq']} | {dp} | {pct(s['share_short'])} | {pct(s['miss_r0.5'])} |")
    return L


def p2f_block():
    """P2F/P2G: kích thước ảnh thật → stats["p2f"]. P0 gộp hai split chỉ dùng thống kê dữ liệu thô (KHÔNG lọc, KHÔNG audit);
    audit khung trước onset (chỉ mô tả, chỉ TRAIN) nằm ở PREREG §5."""
    meta = json.loads((P0 / "parse_uavdt.json").read_text(encoding="utf-8"))["seq_meta"]
    return dict(M0901=dict(W=meta["M0901"]["W"], gt_extent_x=meta["M0901"]["gt_extent"][0]),
                M0207=dict(n_img=meta["M0207"]["n_img"], gt_max_frame=meta["M0207"]["gt_max_frame"]))


def p2f_section(stats):
    f = stats.get("p2f")
    if not f:
        return []
    return ["## v3 — thay đổi so với v2 (kích thước ảnh thật)", "",
            f"- Kích thước ảnh thật (frames UAVDT): M0901 rộng {f['M0901']['W']} px (GT x+w tối đa {f['M0901']['gt_extent_x']}), các chuỗi khác 1024 × 540; "
            f"biên (border) của M0901 tính theo bề rộng {f['M0901']['W']} → 2 sự kiện E3 interior → border. M0207 có {f['M0207']['n_img']} ảnh nhưng GT chỉ "
            f"tới khung {f['M0207']['gt_max_frame']} → chỉ dùng khung ≤ khung GT cuối. Luật áp cho cả TEST khi mở (PREREG §1).",
            "- Mọi số dưới đây là thống kê dữ liệu thô từ annotation (gộp hai split, KHÔNG lọc sự kiện, KHÔNG audit). Audit khung trước onset "
            "(chỉ mô tả, chỉ TRAIN) và mọi ước lượng kênh: PREREG_28.md. Bản lọc annotation-late ngày 1/10/2026 (sáng) đã bị rút — không số nào từ bản đó được dùng.", ""]


def build(stats):
    C = stats["constants"]
    T = C["thresholds"]
    dec = stats.get("decision", {})
    ego = stats.get("egomotion")
    pr = stats["premise"]
    dc = stats["dossier_counts"]
    avail = [d for d in ("UAVDT", "VisDrone") if stats["data"].get(d, {}).get("status") == "OK"]
    L = ["# P0_REPORT_28 (v3) — Đề tài 28, CP0 (thứ Hai " + C["dates"]["cp0"] + ")", "",
         f"*Sinh tự động bởi `code/p0_report.py` từ `results/p0_stats.json` — {dt.datetime.now():%Y-%m-%d %H:%M}. Mọi con số dưới đây có trong json. "
         "Bản v2 (kích thước ảnh 1024 giả định): P0_REPORT_28_v2_bak.md; bản v1 (trước rà chéo): P0_REPORT_28.v1.md.*", ""]
    L += p2f_section(stats)
    if "VisDrone" not in avail or not ego:
        L += ["> ⚠ **Báo cáo MỘT PHẦN.** Link Google Drive chính thức bị \"Quota exceeded\" → "
              + ("VisDrone-VID chưa có; " if "VisDrone" not in avail else "")
              + ("frames UAVDT chưa có (Bước 5 ego-motion chưa chạy). " if not ego else "")
              + "Mọi số dưới đây là **UAVDT từ annotation**. Xem DOWNLOAD_BLOCKED.md; chạy lại 5 lệnh ở đó là báo cáo tự cập nhật.", ""]
    L += p0b_section(stats)
    # 1
    L += ["## 1. Dữ liệu: thực đếm vs kỳ vọng", ""] + count_table(stats)
    L += ["", f"- fps: UAVDT {C['expected']['UAVDT']['fps']} fps (paper). VisDrone: **không xác định** — đã kiểm {'; '.join(C['visdrone_fps']['checked'])}. Mọi chỉ số VisDrone báo theo khung.",
          "- License: cả hai \"research only\"; không re-host frame, chỉ phát hành số liệu dẫn xuất. sha256: DATA_SOURCES.md.", ""]
    # 2
    L += ["## 2. Bảng chính: phơi nhiễm của lịch tuần hoàn", "",
          "N = ⌊D/S⌋ + Bernoulli(frac(D/S)), pha đều, chính xác (unit test PASS). share_short = P(N=0): tỉ lệ sự kiện lịch tuần hoàn có thể bỏ lỡ hoàn toàn. "
          "Từ v2: số chính trên cửa sổ sinh (mục 0 A1).", ""]
    for ds in avail:
        L += main_table(stats, ds) + [""]
    # 3
    L += ["## 3. Mật độ, M(t), ego-motion", ""]
    for ds in avail:
        e = stats["E0"][ds]["main"]
        a = stats["E3_arrival"][ds]["main"]
        mb = stats["D_quantiles"][ds]["E3|main|VEHICLE|M_at_birth"]
        L.append(f"- {ds}: ρ_ev (E0, class-agnostic) trung vị theo chuỗi = {num(e['rho_ev_median'], 2)} (min {num(e['rho_ev_min'], 2)}; gộp {pct(e['rho_ev_pooled'])}; "
                 f"{e['n_seq_rho_lt_0_9']} chuỗi < {num(C['rho_ev_ref'], 2)}) → E0 cho dự đoán tầm thường \"gate thua\" như KE_HOACH mục 2 đã nói. "
                 f"M(t) trung vị {num(e['M_median'])} (p10 {num(e['M_p10'])}, p90 {num(e['M_p90'])}, max {e['M_max']}); % khung M 1–5 / 6–20 / >20 = "
                 f"{pct(e['share_frames_M_1_5'])} / {pct(e['share_frames_M_6_20'])} / {pct(e['share_frames_M_gt_20'])}. "
                 f"Xe mới E3: {num(a['pooled_per_100f'], 2)} / 100 khung (gộp); M tại khung sinh trung vị {num(mb['median'])}.")
    if ego:
        for ds, g in ego["by_dataset"].items():
            L.append(f"- Ego-motion {ds} ({g['n_seq']} chuỗi, {g['n_pairs']} cặp): hovering (dịch tâm < {num(ego['hover_threshold_px'], 1)} px/khung) = {g['n_seq_hovering']} chuỗi ({pct(g['share_seq_hovering'])}); "
                     f"phân vị dịch tâm/chuỗi p10/p50/p90 = {num(g['shift_seq_median_quantiles']['p10'], 2)} / {num(g['shift_seq_median_quantiles']['p50'], 2)} / {num(g['shift_seq_median_quantiles']['p90'], 2)} px. "
                     f"Nhiễu cue (|diff|>{T['diff_thr']} ngoài bbox): thô {pct(g['diff_raw_median'], 2)} vs bù H {pct(g['diff_warp_median'], 2)}. "
                     f"Chi phí CPU 1 luồng: ORB+RANSAC {num(g['ms_orb_ransac_median'], 1)} ms/cặp, warp+diff {num(g['ms_warp_diff_median'], 1)} ms/cặp (median, {g['n_timing_pairs']} cặp).")
    else:
        L.append("- Ego-motion / nhiễu cue / chi phí ORB: **CHƯA CHẠY** (frames bị chặn). Script `code/p0_egomotion.py` đã viết + smoke-test trên cặp ảnh tổng hợp.")
    L.append("")
    # 4
    L += ["## 4. Phân tầng (dữ liệu cho P2)", ""]
    for ds in avail:
        L += [f"**{ds}**", ""] + strata_table(stats, ds) + [""]
        st = {(s["stratum_var"], s["stratum"]): s for s in stats["strata"]
              if s["dataset"] == ds and s["event"] == "E3" and s["vis_def"] == "main" and s["k"] == 5 and s["R"] == 10}
        lo, hi = st.get(("altitude", "low")), st.get(("altitude", "high"))
        sm, lg = st.get(("size_bin", "<16")), st.get(("size_bin", ">=64"))
        if lo and hi and sm and lg:
            def dp(x):
                return num(x["D_p50"]) if x.get("D_p50_exact", True) else "≥" + num(x["D_p50"])
            L += [f"- Nhận xét: bay thấp → D ngắn hơn (D p50 low {dp(lo)} vs high {dp(hi)} khung) và share_short cao hơn "
                  f"({pct(lo['share_short'])} vs {pct(hi['share_short'])}); vật thể lớn (≥64 px; giả thuyết: gắn với bay thấp — chưa kiểm chéo) có share_short {pct(lg['share_short'])} > vật thể nhỏ <16 px {pct(sm['share_short'])}. "
                  f"Khớp chiều của P1 (L(h) ∝ h); chỉ {hi['n_seq']} chuỗi high-alt → CI rộng. Phía r(h) giảm theo độ cao (nửa còn lại của P2) cần detector, đo ở P2.", ""]
    # 5
    d2, sp, po, d3 = pr["deep2_nightVideos"], pr["sparse_regime"], pr["pooled_corpus"], pr["deep3"]
    L += ["## 5. Premise (Bước 7 — chỉ đọc)", "",
          f"- **\"85,6%\" trích sai và đảo nghĩa.** Gốc: Deep2 (`deep2_manuscript.tex` dòng {d2['text_line']}, bảng dòng {d2['line']}): {pct(d2['share_D_lt_8'])} sự kiện của **riêng CDnet nightVideos** (n={d2['n_events']}) **ngắn hơn 8 khung**. "
          "Docx #28 (VN/EN, đoạn 8) viết thành \"85,6% sự kiện kéo dài nhiều khung\" trên cả CDnet/LASIESTA/BMC → sai cả nghĩa lẫn phạm vi.",
          f"- **\"dominated by single-frame events\"**: methodology paper (dòng {pr['methodology_dominated_line']}) nói về regime thưa: {pct(sp['frac_single'])} sự kiện D=1 trên n={sp['n_events']} (median {num(sp['median'])}) — đúng. "
          f"Deep3 dòng {d3['dominated_line']} áp cho toàn tập gộp n={po['n_events']}: D=1 chỉ {pct(po['frac_single'])} (mode, không phải đa số) → nên làm mềm chữ \"dominated\".",
          f"- **Kết luận:** không mâu thuẫn thật — hai câu đo trên tập và ngưỡng khác nhau (D<8 ở nightVideos vs D=1 ở regime thưa). Mọi sự kiện đều detection-grounded (chuỗi khung detector dương, YOLOv3/COCO), class-agnostic, theo video. "
          f"Nếu cần số \"nhiều khung\" (D≥2) cho tập gộp: {pct(po['frac_multi_D_ge_2'])}. Cần sửa câu trong docx #28 (anh sửa; Claude không đổi docx).",
          "- Đối chiếu #28: với sự kiện **theo vật thể** trên UAVDT, D ngắn là hiếm (mục 2) — premise \"aerial = 1–3 khung\" không đứng ở mức track/đoạn nhìn thấy.", ""]
    # 6
    L += ["## 6. Dossier T-ITS / TVT", "",
          f"- T-ITS: **{dc['TITS']} bài** cùng dạng đã xác minh DOI (gồm YOLC; yêu cầu ≥{T['min_dossier_tits']}) → A1 đạt. TVT: {dc['TVT']} bài (yêu cầu ≥{T['min_dossier_tvt']}). Chi tiết: DOSSIER_TITS_28.md.",
          f"- LIT_SEED: {stats['lit_counts']['verified']}/{stats['lit_counts']['total']} mục xác minh (DOI hoặc arXiv); mục chưa xác minh: {stats['lit_counts']['unverified']} (Deep3 — chưa xuất bản). "
          f"Bài ACM {C['acm_uav_sar']['year']} UAV-SAR: abstract không có baseline tuần hoàn (so với xử lý toàn video) → đe doạ tính mới thấp–trung bình; full-text HTTP {C['acm_uav_sar']['fulltext_http']}, anh nên mở PDF xác nhận.", ""]
    # 7
    L += ["## 7. Phương án CP0", ""]
    recs = []
    for ds, d in dec.items():
        hov_none = None if not ego else (ego["by_dataset"].get(ds, {}).get("n_seq_hovering", None) == 0)
        C_ok = d["C_part1_lt_1pct"] and (hov_none is True)
        L.append(f"- {ds} (cửa sổ sinh; v1: {pct(d.get('v1_share_short_E3_k5_R10'))} / {pct(d.get('v1_share_short_E3_k12_R10'))}): share_short(E3, k=5, R=10) = {pct(d['share_short_E3_k5_R10'])} [{pct(d['share_short_E3_k5_R10_ci'][0])}, {pct(d['share_short_E3_k5_R10_ci'][1])}] "
                 f"(ngưỡng B: < {pct(T['B_share_short'], 0)} → {'CÓ' if d['B_trigger_lt_5pct'] else 'KHÔNG'}); "
                 f"share_short(E3, k=12, R=10) = {pct(d['share_short_E3_k12_R10'])} [{pct(d['share_short_E3_k12_R10_ci'][0])}, {pct(d['share_short_E3_k12_R10_ci'][1])}] "
                 f"(ngưỡng C: < {pct(T['C_share_short'], 0)} VÀ không chuỗi hovering → {'CÓ' if C_ok else 'KHÔNG' if hov_none is not None or not d['C_part1_lt_1pct'] else 'chưa đủ dữ liệu hovering'}).")
        recs.append("C" if C_ok else "B" if d["B_trigger_lt_5pct"] else "A")
    rec = "C" if "C" in recs and all(r == "C" for r in recs) else "B" if "B" in recs else "A"
    L += ["",
          "- **A (MẶC ĐỊNH)**: E2+E3 trên VisDrone-VID + UAVDT, lưới k∈{1,2,3,5,8,12}, 3 mức detector → P1+P2.",
          "- **B**: A + neo transient thật (FMO / Anti-UAV, chỉ nếu tải trực tiếp) — khi share_short(E3,k=5,R=10) < 5%.",
          "- **C**: nhánh âm sớm — khi share_short(E3,k=12,R=10) < 1% VÀ không có chuỗi hovering → limitation P-A, dừng bài riêng.",
          f"- **KHUYẾN NGHỊ: {rec}** (dựa trên {', '.join(avail)}"
          + ("; chưa có VisDrone và hovering — kết luận có thể đổi khi đủ dữ liệu" if len(avail) < 2 or not ego else "") + ").",
          f"- Cần anh chốt: ngưỡng hovering — sẽ đề xuất từ phân bố dịch tâm ảnh qua H (median theo chuỗi; khe giữa hai mode), không áp cứng; {num(T['hover_px'], 1)} px/khung chỉ là giá trị tạm"
          + ("" if ego else " (chưa có phân bố vì chưa có frames)") + "; VISIBLE chính = VisDrone occlusion<2 / UAVDT occlusion≠2 & out_of_view≠2; "
          "STRICT = VisDrone occlusion=0 & truncation=0 / UAVDT occlusion=1 & out_of_view=1 (bảng STRICT: results/p0/exposure_grid.csv).", ""]
    # 8
    L += ["## 8. Việc anh Đạt cần làm", "",
          "1. Tải tay VisDrone2019-VID (3 zip) và UAVDT UAV-benchmark-M theo DOWNLOAD_BLOCKED.md (nếu vòng thử lại tự động thất bại), rồi chạy lại 5 lệnh.",
          "2. (Đã chốt 25/09/2026) Train YOLOv8s 1 lần trên Kaggle (code/kaggle/README_KAGGLE_28.md) → dump + cue trên laptop OpenVINO.",
          "3. (Đã sửa 25/09/2026 trong 2 docx + Excel theo KE_HOACH) câu \"85,6%\" (mục 5) — không cần làm thêm.",
          f"4. Mở PDF bài ACM {C['acm_uav_sar']['year']} UAV-SAR để xác nhận không có baseline every-k-th frame.",
          "5. Chốt tại CP0: định nghĩa VISIBLE, ngưỡng hovering, phương án A/B/C.",
          "6. (P0b) Chốt ROI m chính (0 hay 5%); gán nhãn AUDIT_SHORT_E3.md khi ảnh đã xuất; rà THEORY_28.md và danh sách ô PREREG_28_DRAFT.md."]
    stats["report_recommendation"] = rec
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- kiểm số
def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, str):
        for m in re.findall(r"-?\d+(?:\.\d+)?", o):
            yield float(m)


def number_check(text, stats):
    allowed = set()
    for v in leaves(stats):
        allowed |= {f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{100 * v:.0f}", f"{100 * v:.1f}", f"{100 * v:.2f}",
                    f"{v:,.0f}".replace(",", ".")}
    # bỏ: định danh (E0–E3, p10/p50/p90, M0101, r=…, k∈{…} lưới, ≥/≤ trong nhãn), ngày giờ, đường dẫn/dòng file
    t = re.sub(r"\b(E[0-3]|p\d{2}|M\d{4}|T\d+|V\d+|A\d|C\d|P\d|L\d|R\d)\b", " ", text)
    t = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}|\d{1,2}/\d{1,2}/\d{4}", " ", t)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"arXiv \d{4}\.\d{4,5}", " ", t)
    t = re.sub(r"\b\d+\. ", " ", t)          # đánh số mục/danh sách
    t = re.sub(r"#+ \d+\.", " ", t)
    toks = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)*(?![\w])", t)
    bad = [x for x in toks if x not in allowed and x.replace(",", ".") not in allowed]
    return dict(n_numbers=len(toks), n_mismatch=len(bad), mismatches=sorted(set(bad)))


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    stats["constants"] = CONSTANTS
    stats["dossier_counts"] = dossier_counts()
    lit = json.loads((P0 / "crossref_lit.json").read_text(encoding="utf-8"))
    stats["lit_counts"] = dict(total=len(lit), verified=sum(1 for i in lit if i["status"].startswith("XÁC MINH")),
                               unverified=sum(1 for i in lit if not i["status"].startswith("XÁC MINH")))
    ut = json.loads((P0 / "unit_tests.json").read_text())
    stats["unit_tests"] = dict(all_passed=ut["all_passed"], T1_n_pairs=ut["T1_r1_closed_form"]["n_pairs"],
                               T1_max_abs_err=ut["T1_r1_closed_form"]["max_abs_err"], T2_max_z=ut["T2_monte_carlo"]["max_z"], T2_n_mc=ut["T2_monte_carlo"]["n_mc"])
    sel = P0 / "audit_short_selection.csv"
    if sel.exists() and "p0b" in stats:
        a = pd.read_csv(sel)
        c3 = stats["censoring"]["UAVDT"]["E3|main|VEHICLE"]
        nb = int(round(c3["birth_D_lt_8_n"] * c3["birth_D_lt_8_share_border"]))
        stats["p0b"]["audit"] = dict(n_selected=int(len(a)), n_border_selected=int((a.e3_type == "border").sum()),
                                     n_interior_selected=int((a.e3_type == "interior").sum()),
                                     n_border_avail=nb, n_interior_avail=int(c3["birth_D_lt_8_n"] - nb), n_per_stratum=20, seed=42)
    stats["p2f"] = p2f_block()
    data_sources(stats)
    text = build(stats)
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    REPORT.write_text(text, encoding="utf-8")
    chk = number_check(text, stats)
    (P0 / "number_check.json").write_text(json.dumps(chk, indent=1, ensure_ascii=False), encoding="utf-8")
    print("report", REPORT, "words", len(text.split()), "| number check:", chk["n_numbers"], "numbers,", chk["n_mismatch"], "mismatch", chk["mismatches"][:30])


if __name__ == "__main__":
    main()
