"""P0 bước 8: xác minh DOI của LIT_SEED qua Crossref + doi.org; truy vấn dossier T-ITS / TVT.

Xuất: results/p0/crossref_lit.json, results/p0/crossref_dossier.json
(LIT_28.md và DOSSIER_TITS_28.md do p0_report.py / tay-chọn dựa trên các json này.)
"""
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p0"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "P0-28-crossref/1.0 (mailto:lamquocdat@gmail.com)"}


def crossref(doi):
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=30)
    except requests.RequestException as e:
        return {"ok": False, "err": str(e)}
    if r.status_code != 200:
        return {"ok": False, "err": f"HTTP {r.status_code}"}
    m = r.json()["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "published"):
        dp = m.get(k, {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            year = dp[0][0]
            break
    return {
        "ok": True,
        "title": (m.get("title") or [""])[0],
        "year": year,
        "venue": (m.get("container-title") or [""])[0],
        "authors": [f"{a.get('family', '')}" for a in m.get("author", [])][:4],
        "type": m.get("type"),
        "abstract": re.sub(r"<[^>]+>", " ", m.get("abstract", "") or "").strip(),
    }


def resolve(doi):
    try:
        r = requests.head(f"https://doi.org/{doi}", allow_redirects=False, timeout=30, headers=UA)
        loc = r.headers.get("Location", "")
        return {"status": r.status_code, "location": loc, "ok": r.status_code in (301, 302, 303, 307, 308) and bool(loc)}
    except requests.RequestException as e:
        return {"status": None, "ok": False, "err": str(e)}


def search_crossref(query_title, n=3):
    """Tìm DOI cho mục 'chưa xác minh' bằng tên bài (chỉ đề xuất; vẫn phải so title)."""
    r = requests.get("https://api.crossref.org/works", headers=UA, timeout=30,
                     params={"query.bibliographic": query_title, "rows": n,
                             "select": "DOI,title,container-title,issued,type"})
    if r.status_code != 200:
        return []
    out = []
    for it in r.json()["message"]["items"]:
        out.append({"doi": it["DOI"], "title": (it.get("title") or [""])[0],
                    "venue": (it.get("container-title") or [""])[0],
                    "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "type": it.get("type")})
    return out


def arxiv(aid):
    try:
        r = requests.get("http://export.arxiv.org/api/query", params={"id_list": aid}, timeout=30, headers=UA)
        t = re.findall(r"<title>(.*?)</title>", r.text, re.S)
        pub = re.findall(r"<published>(\d{4})", r.text)
        doi = re.findall(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", r.text)
        if len(t) >= 2:
            return {"ok": True, "title": " ".join(t[1].split()), "year": int(pub[0]) if pub else None,
                    "doi_in_arxiv": doi[0] if doi else None}
        return {"ok": False, "err": "not found"}
    except requests.RequestException as e:
        return {"ok": False, "err": str(e)}


def parse_seed():
    txt = (ROOT / "LIT_SEED_28_20260925.md").read_text(encoding="utf-8")
    items = []
    for line in txt.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if not m:
            continue
        num, body = int(m.group(1)), m.group(2)
        dois = re.findall(r"\b(10\.\d{4,9}/[^\s,;)]+)", body)
        arx = re.findall(r"arXiv:(\d{4}\.\d{4,5})", body)
        items.append({"n": num, "cite": body, "dois": [d.rstrip(".") for d in dois], "arxiv": arx,
                      "seed_unverified": "chưa xác minh" in body})
    return items


# tiêu đề dùng để tìm DOI cho các mục seed không ghi DOI
TITLE_QUERIES = {
    4: "Bandwidth-Efficient Live Video Analytics for Drones via Edge Computing",
    5: "FrameHopper: Selective Processing of Video Frames in Detection-driven Real-Time Video Analytics",
    7: "Looking Fast and Slow: Memory-Guided Mobile Video Object Detection",
    17: "Detection and Tracking Meet Drones Challenge",
    18: "The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking",
    19: "The World of Fast Moving Objects",
    22: "Need for Speed: A Benchmark for Higher Frame Rate Object Tracking",
    23: "Towards Streaming Perception",
    24: "PVT++: A Simple End-to-End Latency-Aware Visual Tracking Framework",
    25: "Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking ColTrack",
    27: "Comparison of Riemann and Lebesgue sampling for first order stochastic systems",
    28: "Sampling of the Wiener Process for Remote Estimation over a Channel with Random Delay",
    20: "Anti-UAV: A Large-Scale Benchmark for Vision-based UAV Tracking",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def title_match(a, b):
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def lit():
    items = parse_seed()
    for it in items:
        it["checks"] = []
        for d in it["dois"]:
            cr = crossref(d)
            rs = resolve(d)
            it["checks"].append({"doi": d, "crossref": cr, "resolve": rs})
            time.sleep(0.3)
        for a in it["arxiv"]:
            it.setdefault("arxiv_checks", []).append({"id": a, **arxiv(a)})
            time.sleep(3)
        if it["n"] in TITLE_QUERIES:
            cands = search_crossref(TITLE_QUERIES[it["n"]])
            for c in cands:
                c["title_match"] = round(title_match(TITLE_QUERIES[it["n"]], c["title"]), 3)
            it["title_search"] = cands
            best = [c for c in cands if c["title_match"] >= 0.85]
            if best and not it["dois"]:
                b = best[0]
                it["suggested_doi"] = b["doi"]
                it["suggested_resolve"] = resolve(b["doi"])
            time.sleep(0.3)
        ok = [c for c in it["checks"] if c["crossref"]["ok"] and c["resolve"]["ok"]]
        if ok:
            it["status"] = "XÁC MINH"
        elif it.get("suggested_doi") and it["suggested_resolve"]["ok"]:
            it["status"] = "XÁC MINH (DOI tìm theo title)"
        elif any(x.get("ok") for x in it.get("arxiv_checks", [])):
            it["status"] = "XÁC MINH (arXiv)"
        else:
            it["status"] = "KHÔNG"
        print(it["n"], it["status"], [c["crossref"].get("title", c["crossref"].get("err"))[:70] for c in it["checks"]])
    (OUT / "crossref_lit.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


DOSSIER_QUERIES = [
    "UAV traffic vehicle detection video", "drone traffic monitoring", "aerial vehicle detection",
    "UAV video vehicle detection efficient", "drone video edge onboard detection",
    "aerial traffic video frame", "UAV vehicle tracking", "unmanned aerial vehicle traffic surveillance",
    "drone vehicle counting", "aerial image small object detection lightweight", "UAV real-time object detection embedded",
]


def dossier():
    res = {}
    for label, issns in (("TITS", ["1524-9050", "1558-0016"]), ("TVT", ["0018-9545", "1939-9359"])):
        seen = {}
        for q in DOSSIER_QUERIES:
            filt = ",".join([f"issn:{i}" for i in issns] + ["from-pub-date:2023-01-01", "type:journal-article"])
            try:
                r = requests.get("https://api.crossref.org/works", headers=UA, timeout=60,
                                 params={"query.bibliographic": q, "filter": filt, "rows": 25,
                                         "select": "DOI,title,container-title,issued,type,subject"})
            except requests.RequestException as e:
                print("ERR", label, q, e)
                continue
            if r.status_code != 200:
                print("HTTP", r.status_code, label, q)
                continue
            for it in r.json()["message"]["items"]:
                d = it["DOI"].lower()
                if d not in seen:
                    seen[d] = {"doi": d, "title": (it.get("title") or [""])[0],
                               "venue": (it.get("container-title") or [""])[0],
                               "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "queries": []}
                seen[d]["queries"].append(q)
            time.sleep(0.5)
        res[label] = list(seen.values())
        print(label, len(seen))
    (OUT / "crossref_dossier_raw.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def verify(dois, name):
    """Xác minh (Crossref + doi.org) danh sách DOI đã chọn tay cho dossier."""
    out = []
    for d in dois:
        out.append({"doi": d, "crossref": crossref(d), "resolve": resolve(d)})
        time.sleep(0.3)
    (OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for o in out:
        print(o["doi"], o["resolve"]["ok"], o["crossref"].get("year"), o["crossref"].get("venue"), "|", o["crossref"].get("title"))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("lit", "all"):
        lit()
    if cmd in ("dossier", "all"):
        dossier()
    if cmd == "verify":
        verify(sys.argv[3:], sys.argv[2])
