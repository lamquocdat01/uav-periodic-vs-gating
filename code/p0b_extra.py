"""P0b (rà chéo 25-09): A2 độ nhạy ROI, A3 iso-miss, A4 onset latency, A5 mật độ cửa sổ khởi phát.

Đầu vào: rows_/frames_/events_E3_<ds>.parquet (p0_parse, p0_events đã có cờ kiểm duyệt).
Chỉ số chính trên CỬA SỔ SINH (start ≤ seq_last − W, W=120) như p0_stats; r<1 báo thêm Kaplan–Meier.

A2  E3-ROI(m): xe E3 (VEHICLE, sinh sau khung đầu) có tâm bbox đi vào ROI = [mW,(1−m)W]×[mH,(1−m)H],
    m ∈ {0, 5%, 10%}; D = số khung liên tiếp (đoạn đầu tiên) tâm nằm trong ROI, trong đoạn visible đầu tiên.
    Xe không bao giờ vào ROI trong đoạn visible đầu → không là sự kiện ở mức m (đếm riêng).
A3  S*(ε) = S lớn nhất sao cho miss(S') ≤ ε với mọi S' ≤ S (S ∈ 1..120); tần số 30/S* Hz, a = 1/S*.
A4  Lịch tuần hoàn pha đều, stride S: L = khoảng từ khung visible đầu tới lần nhìn THÀNH CÔNG đầu;
    P(L ≤ Lmax) = 1 − miss_r(min(D, Lmax+1), S)  (Lemma 1 áp cho cửa sổ min(D, Lmax+1)).
    S_lat*(p) = S lớn nhất sao cho P(L ≤ Lmax) ≥ p với mọi S' ≤ S.
A5  ρ_onset(w) = tỉ lệ khung nằm trong ≥1 cửa sổ [onset, onset+w) của sự kiện E3 (mọi sự kiện, cắt ở cuối chuỗi).
CI: bootstrap theo chuỗi B=1000 seed=42 (cùng trọng số với p0_stats).
Xuất: results/p0/roi_events_<ds>.parquet, roi_grid.csv, isomiss.csv, latency.csv, rho_onset_seq.csv;
      khối "p0b" trong results/p0_stats.json.
Chạy: python code/p0b_extra.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from p0_events import visible_runs  # noqa: E402
from p0_stats import (B, FPS, KM, SEED, W_BIRTH, birth_window, boot_weights, censor_flags,  # noqa: E402
                      dq, miss_r, p_zero)

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results" / "p0"
STATS = ROOT / "results" / "p0_stats.json"
DATASETS = {"uavdt": "UAVDT", "visdrone": "VisDrone"}

MARGINS = [0.0, 0.05, 0.10]
ROI_KR = [(k, R) for k in (1, 5, 12) for R in (5, 10)]
EPS = [0.10, 0.05, 0.02, 0.01]
R_RECALL = [1.0, 0.8, 0.5]
S_SCAN = np.arange(1, W_BIRTH + 1)
LMAX = [3, 5, 10, 30]
S_LAT = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50, 60, 120]
P_LAT = [0.95, 0.99]
W_ONSET = [3, 5, 10]


# ---------------------------------------------------------------- A2: sự kiện ROI
def roi_events(tag, e3, W_img, H_img):
    """Mỗi (track E3 main, m) → start_roi, D_roi. Dùng đoạn visible đầu tiên (như E3)."""
    rows = pd.read_parquet(P0 / f"rows_{tag}.parquet", columns=["seq", "tid", "frame", "cx", "cy", "visible", "group"])
    rows = rows[rows.group == "VEHICLE"].sort_values(["seq", "tid", "frame"])
    e3 = e3.set_index(["seq", "track_id"])
    keys = set(e3.index)
    recs = []
    for (seq, tid), d in rows.groupby(["seq", "tid"], sort=False):
        if (seq, tid) not in keys:
            continue
        ev = e3.loc[(seq, tid)]
        a, b = int(ev.start), int(ev.start + ev.D - 1)
        dd = d[(d.frame >= a) & (d.frame <= b)]
        Wi, Hi = W_img[seq], H_img[seq]
        for m in MARGINS:
            inside = ((dd.cx >= m * Wi) & (dd.cx <= (1 - m) * Wi) & (dd.cy >= m * Hi) & (dd.cy <= (1 - m) * Hi)).values
            runs = visible_runs(dd.frame.values, inside)
            base = dict(dataset=ev.dataset, split=ev.split, seq=seq, track_id=int(tid), m=m, e3_type=ev.e3_type,
                        seq_last=int(ev.seq_last), seq_first=int(ev.seq_first), e3_start=a, e3_D=int(ev.D))
            if runs:
                s0, s1 = runs[0]
                recs.append(dict(base, start=int(s0), D=int(s1 - s0 + 1), enters_roi=True,
                                 right_censored=bool(s1 >= ev.seq_last), left_censored=False))
            else:
                recs.append(dict(base, start=-1, D=0, enters_roi=False, right_censored=False, left_censored=False))
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- tiện ích bootstrap
def seq_sums(df, seqs, values):
    idx = pd.Categorical(df.seq, categories=seqs).codes
    return (np.bincount(idx, weights=values, minlength=len(seqs)),
            np.bincount(idx, minlength=len(seqs)).astype(float))


def boot_mean(df, seqs, values, Wk):
    """Trung bình theo sự kiện với trọng số chuỗi; Wk hàng 0 = ước lượng điểm. Trả vector (1+B)."""
    s, n = seq_sums(df, seqs, values)
    return (Wk @ s) / np.maximum(Wk @ n, 1e-12)


def ci(v):
    return [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]


def km_of(df, seqs):
    codes = pd.Categorical(df.seq, categories=seqs).codes
    return KM(df.D.values, ~df.right_censored.values, codes, len(seqs))


def max_ok(curve_ok):
    """curve_ok: (…, len(S)) bool theo S tăng dần → S lớn nhất mà mọi S' ≤ S đều ok; 0 nếu S=1 đã hỏng."""
    bad = ~curve_ok
    first_bad = np.where(bad.any(axis=-1), bad.argmax(axis=-1), curve_ok.shape[-1])
    return first_bad  # = số S liên tiếp từ 1 thoả → chính là S* vì S_SCAN bắt đầu từ 1


def s_star_summary(Sstar, fps):
    pt = int(Sstar[0])
    lo, hi = (int(np.percentile(Sstar[1:], 2.5)), int(np.percentile(Sstar[1:], 97.5)))
    out = dict(S_star=pt, S_star_ci=[lo, hi], capped=bool(pt >= S_SCAN[-1]), infeasible=bool(pt == 0))
    out["a_star"] = (1.0 / pt) if pt else None
    out["a_star_ci"] = [1.0 / hi if hi else None, 1.0 / lo if lo else None]
    if fps:
        out["hz_star"] = (fps / pt) if pt else None
        out["hz_star_ci"] = [fps / hi if hi else None, fps / lo if lo else None]
    return out


# ---------------------------------------------------------------- A5
def rho_onset(fr, e3, seqs, w, start_col="start"):
    recs = []
    for seq, g in fr.groupby("seq"):
        f0, f1 = int(g.frame.min()), int(g.frame.max())
        cov = np.zeros(f1 - f0 + 1, bool)
        for s in e3.loc[e3.seq == seq, start_col].values:
            if s < f0:
                continue
            cov[int(s) - f0: min(int(s) - f0 + w, len(cov))] = True
        recs.append(dict(seq=seq, w=w, n_frames=len(cov), n_cov=int(cov.sum()), rho=float(cov.mean()),
                         n_events=int((e3.seq == seq).sum())))
    return pd.DataFrame(recs)


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    out = dict(meta=dict(margins=MARGINS, roi_kr=ROI_KR, eps=EPS, r=R_RECALL, S_scan_max=int(S_SCAN[-1]), Lmax=LMAX, S_lat=S_LAT,
                         p_lat=P_LAT, w_onset=W_ONSET, W_birth=W_BIRTH, B=B, seed=SEED,
                         roi_def="tâm bbox trong [mW,(1−m)W]×[mH,(1−m)H]; D = đoạn liên tiếp đầu tiên trong ROI thuộc đoạn visible đầu",
                         iso_def="S* = max S: miss(S') ≤ ε ∀ S' ≤ S, S ∈ 1..120",
                         lat_def="P(L ≤ Lmax) = 1 − miss_r(min(D, Lmax+1), S)"),
               roi={}, isomiss={}, latency={}, rho_onset={})
    roi_rows, iso_rows, lat_rows, rho_rows = [], [], [], []
    for tag, ds in DATASETS.items():
        if stats["data"].get(ds, {}).get("status") != "OK":
            continue
        fps = FPS[ds]
        fr = pd.read_parquet(P0 / f"frames_{tag}.parquet")
        seqs = sorted(fr.seq.unique())
        Wb = boot_weights(seqs)
        Wk = np.vstack([np.ones(len(seqs)), Wb])
        last, first = fr.groupby("seq").frame.max(), fr.groupby("seq").frame.min()
        e3 = pd.read_parquet(P0 / f"events_E3_{tag}.parquet")
        e3 = e3[(~e3.never_visible) & (e3.vis_def == "main")]
        e3 = censor_flags(e3, last, first)
        meta = json.loads((P0 / f"parse_{tag}.json").read_text(encoding="utf-8"))["seq_meta"]
        W_img = {s: m["W"] for s, m in meta.items()}
        H_img = {s: m["H"] for s, m in meta.items()}

        # ---------------- A2
        ev = roi_events(tag, e3, W_img, H_img)
        ev.to_parquet(P0 / f"roi_events_{tag}.parquet", index=False)
        out["roi"][ds] = {}
        n_short_m0 = None
        populations = {}
        for m in MARGINS:
            em = ev[(ev.m == m) & ev.enters_roi]
            eb = birth_window(em)
            populations[m] = (em, eb)
            km = km_of(em, seqs)
            short = eb[eb.D < 8]
            if m == 0.0:
                n_short_m0 = len(short)
            q = dq(eb.D.values, fps)
            q.update({f"p{p}_exact": bool(q[f"p{p}"] <= W_BIRTH) for p in (10, 50, 90)})
            kmq = {}
            for p in (10, 50, 90):
                v, ok = km.quantile(p / 100)
                kmq[f"p{p}"], kmq[f"p{p}_reached"] = v, ok
            rec = dict(n_e3=int((ev.m == m).sum()), n_enter=int(len(em)), n_not_enter=int((~ev[ev.m == m].enters_roi).sum()),
                       n_birth=int(len(eb)), n_birth_right_censored=int(eb.right_censored.sum()),
                       D_birth=q, D_km=kmq,
                       D_lt_8_n=int(len(short)), D_lt_8_share=float(len(short) / len(eb)),
                       D_lt_8_border_n=int((short.e3_type == "border").sum()), D_lt_8_interior_n=int((short.e3_type == "interior").sum()),
                       D_lt_8_remaining_vs_m0=float(len(short) / n_short_m0) if n_short_m0 else None,
                       by_type={t: dict(n=int((eb.e3_type == t).sum()),
                                        D_lt_8_share=float((eb[eb.e3_type == t].D < 8).mean()) if (eb.e3_type == t).any() else None)
                                for t in ("border", "interior")},
                       cells={})
            for k, R in ROI_KR:
                S = k * R
                v = boot_mean(eb, seqs, p_zero(eb.D.values, S), Wk)
                c = dict(k=k, R=R, S=S, share_short=float(v[0]), share_short_ci=ci(v))
                for r in (0.8, 0.5):
                    vb = boot_mean(eb, seqs, miss_r(eb.D.values, S, r), Wk)
                    vk = km.expect(lambda T, r=r: miss_r(T, S, r), Wk)
                    c[f"miss_r{r}_birth"], c[f"miss_r{r}_birth_ci"] = float(vb[0]), ci(vb)
                    c[f"miss_r{r}_km"], c[f"miss_r{r}_km_ci"] = float(vk[0]), ci(vk)
                rec["cells"][f"S{S}"] = c
                roi_rows.append(dict(dataset=ds, m=m, **{kk: vv for kk, vv in c.items() if not kk.endswith("_ci")},
                                     share_short_lo=c["share_short_ci"][0], share_short_hi=c["share_short_ci"][1], n_birth=len(eb)))
            out["roi"][ds][f"m{m:.2f}"] = rec

        # ---------------- A3 iso-miss (m ∈ {0, 5%})
        out["isomiss"][ds] = {}
        for m in (0.0, 0.05):
            em, eb = populations[m]
            km = km_of(em, seqs)
            # đường cong miss(S) (1+B) × |S|
            for r in R_RECALL:
                curves = {"birth": np.stack([boot_mean(eb, seqs, miss_r(eb.D.values, S, r), Wk) for S in S_SCAN], axis=1)}
                if r < 1.0:
                    curves["km"] = np.stack([km.expect(lambda T, S=S: miss_r(T, S, r), Wk) for S in S_SCAN], axis=1)
                for est, cur in curves.items():
                    key = f"m{m:.2f}|r{r}|{est}"
                    out["isomiss"][ds][key] = {"miss_curve_point": {int(S): float(cur[0, i]) for i, S in enumerate(S_SCAN) if S in (1, 2, 3, 5, 10, 14, 15, 30, 60, 120)}}
                    for eps in EPS:
                        Sst = max_ok(cur <= eps + 1e-12)
                        sm = s_star_summary(Sst, fps)
                        out["isomiss"][ds][key][f"eps{eps}"] = sm
                        iso_rows.append(dict(dataset=ds, m=m, r=r, estimator=est, eps=eps, **{kk: vv for kk, vv in sm.items() if not kk.endswith("_ci")},
                                             S_star_lo=sm["S_star_ci"][0], S_star_hi=sm["S_star_ci"][1]))

        # ---------------- A4 onset latency (m ∈ {0, 5%})
        out["latency"][ds] = {}
        for m in (0.0, 0.05):
            _, eb = populations[m]
            for r in R_RECALL:
                for Lm in LMAX:
                    T = np.minimum(eb.D.values, Lm + 1)  # ≤ 31 < W+1 ⇒ không bị kiểm duyệt trong cửa sổ sinh
                    key = f"m{m:.2f}|r{r}|L{Lm}"
                    grid = {}
                    for S in S_LAT:
                        v = boot_mean(eb, seqs, 1.0 - miss_r(T, S, r), Wk)
                        grid[f"S{S}"] = dict(p=float(v[0]), ci=ci(v))
                        lat_rows.append(dict(dataset=ds, m=m, r=r, Lmax=Lm, S=S, p_L_le=float(v[0]), lo=ci(v)[0], hi=ci(v)[1]))
                    cur = np.stack([boot_mean(eb, seqs, 1.0 - miss_r(T, S, r), Wk) for S in S_SCAN], axis=1)
                    need = {}
                    for pt in P_LAT:
                        need[f"p{pt}"] = s_star_summary(max_ok(cur >= pt - 1e-12), fps)
                    out["latency"][ds][key] = dict(grid=grid, S_needed=need, Lmax_s=(Lm / fps if fps else None))

        # ---------------- A5 ρ_onset
        out["rho_onset"][ds] = {}
        for m in (0.0, 0.05):
            em, _ = populations[m]
            for w in W_ONSET:
                rs = rho_onset(fr, em, seqs, w)
                rs["m"], rs["dataset"] = m, ds
                rho_rows.append(rs)
                s_cov = rs.set_index("seq").reindex(seqs).n_cov.values.astype(float)
                s_n = rs.set_index("seq").reindex(seqs).n_frames.values.astype(float)
                pooled = (Wk @ s_cov) / (Wk @ s_n)
                out["rho_onset"][ds][f"m{m:.2f}|w{w}"] = dict(
                    pooled=float(pooled[0]), pooled_ci=ci(pooled), seq_median=float(rs.rho.median()),
                    seq_p10=float(rs.rho.quantile(.1)), seq_p90=float(rs.rho.quantile(.9)), seq_max=float(rs.rho.max()),
                    n_seq_zero=int((rs.rho == 0).sum()), lambda_per_frame=float(rs.n_events.sum() / rs.n_frames.sum()),
                    lambda_w_naive=float(w * rs.n_events.sum() / rs.n_frames.sum()))
    pd.DataFrame(roi_rows).to_csv(P0 / "roi_grid.csv", index=False)
    pd.DataFrame(iso_rows).to_csv(P0 / "isomiss.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(P0 / "latency.csv", index=False)
    if rho_rows:
        pd.concat(rho_rows, ignore_index=True).to_csv(P0 / "rho_onset_seq.csv", index=False)
    stats["p0b"] = out
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    ds = "UAVDT"
    if ds in out["roi"]:
        for m, r in out["roi"][ds].items():
            print("ROI", m, r["n_enter"], r["n_birth"], "D<8", r["D_lt_8_n"], f"{r['D_lt_8_share']:.3f}", "S50", round(r["cells"]["S50"]["share_short"], 4),
                  "S120", round(r["cells"]["S120"]["share_short"], 4))
        for k, v in out["isomiss"][ds].items():
            print("ISO", k, {e: (v[f"eps{e}"]["S_star"], v[f"eps{e}"]["S_star_ci"]) for e in EPS})
        for k, v in out["latency"][ds].items():
            print("LAT", k, {p: (s["S_star"], s["S_star_ci"]) for p, s in v["S_needed"].items()})
        for k, v in out["rho_onset"][ds].items():
            print("RHO", k, round(v["pooled"], 4), [round(x, 4) for x in v["pooled_ci"]], round(v["seq_median"], 4))


if __name__ == "__main__":
    main()
