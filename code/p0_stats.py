"""P0 bước 4: mức độ phơi nhiễm của lịch detector tuần hoàn (chính xác, không Monte Carlo).

Decimation k rồi detector mỗi R khung giữ lại ≡ lịch tuần hoàn stride S=k·R trên khung gốc, pha đều.
Sự kiện D khung gốc: N = floor(D/S) + Bernoulli(frac(D/S)).
  share_short = P(N=0) = (1 - D/S)_+
  miss_r      = E[(1-r)^N] = (1-f)(1-r)^n + f(1-r)^(n+1),  n=floor(D/S), f=frac(D/S)
Trung bình theo sự kiện; CI = bootstrap theo CHUỖI (B=1000, seed=42, percentile 95%).

P0b-A1 (sửa kiểm duyệt phải, rà chéo 25-09): cột `window`
  birth_W120  CHÍNH: chỉ sự kiện có start ≤ seq_last − W (W = S_max = 120) và không kiểm duyệt trái.
              r=1: CHÍNH XÁC vì sự kiện bị cắt có D_obs ≥ W+1 > S ⇒ (1−D/S)_+ = 0 như D thật.
              r<1: CẬN TRÊN (miss_r không tăng theo D, D_obs ≤ D).
  all_v1      như P0 v1 (mọi sự kiện, D bị cắt coi như D thật) — chỉ để so số cũ → mới.
  km          Kaplan–Meier cho F_D (mọi sự kiện không kiểm duyệt trái; cắt ở khung cuối = censored);
              khối lượng đuôi còn lại gán tại thời gian lớn nhất (bảo thủ cho hàm giảm theo D).

Xuất results/p0_stats.json (mọi con số dùng trong báo cáo) + results/p0/exposure_grid.csv, strata.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results" / "p0"
STATS = ROOT / "results" / "p0_stats.json"

K_GRID = [1, 2, 3, 5, 8, 12]
R_GRID = [2, 3, 4, 5, 7, 10, 20]
R_RECALL = [1.0, 0.8, 0.5]
CI_K, CI_R = [1, 3, 5, 12], [5, 10]
B, SEED = 1000, 42
W_BIRTH = 120  # = S_max của lưới (k=12, R=10)
WINDOWS = ("birth_W120", "all_v1", "km")
FPS = {"UAVDT": 30.0, "VisDrone": None}
DATASETS = {"uavdt": "UAVDT", "visdrone": "VisDrone"}


def n_frac(D, S):
    D = np.asarray(D, dtype=np.int64)
    n = D // S
    f = (D - n * S) / S
    return n, f


def p_zero(D, S):
    n, f = n_frac(D, S)
    return np.where(n == 0, 1.0 - f, 0.0)


def miss_r(D, S, r):
    n, f = n_frac(D, S)
    q = 1.0 - r
    if r == 1.0:  # 0^0 = 1
        return np.where(n == 0, 1.0 - f, 0.0)
    return (1 - f) * q ** n + f * q ** (n + 1)


def expected_N(D, S):
    return np.asarray(D) / S


def metrics(D, S):
    out = {"share_short": p_zero(D, S)}
    for r in R_RECALL:
        out[f"miss_r{r}"] = miss_r(D, S, r)
    return out


def boot(df, S, seqs, W):
    """Bootstrap theo chuỗi: W (B × n_seq) số lần chọn mỗi chuỗi. Trả dict metric -> (lo, hi)."""
    res = {}
    D = df.D.values
    seq_idx = pd.Categorical(df.seq, categories=seqs).codes
    n_s = np.bincount(seq_idx, minlength=len(seqs)).astype(float)
    for name, v in metrics(D, S).items():
        s_s = np.bincount(seq_idx, weights=v, minlength=len(seqs))
        stat = (W @ s_s) / np.maximum(W @ n_s, 1e-12)
        res[name] = [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]
    return res


def boot_weights(seqs, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seqs), size=(B, len(seqs)))
    W = np.zeros((B, len(seqs)))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=len(seqs))
    return W


def dq(D, fps):
    q = {f"p{p}": float(np.percentile(D, p)) for p in (10, 50, 90)}
    q["mean"] = float(np.mean(D))
    q["n"] = int(len(D))
    q["share_D_le_1"] = float(np.mean(np.asarray(D) <= 1))
    q["share_D_lt_8"] = float(np.mean(np.asarray(D) < 8))
    if fps:
        q.update({f"p{p}_s": float(np.percentile(D, p) / fps) for p in (10, 50, 90)})
    return q


def size_bin(x):
    return pd.cut(x, [-np.inf, 16, 32, 64, np.inf], right=False, labels=["<16", "16-32", "32-64", ">=64"]).astype(str)


def M_bin(x):
    return pd.cut(x, [-np.inf, 0.5, 5.5, 20.5, np.inf], labels=["0", "1-5", "6-20", ">20"]).astype(str)


def hovering_map():
    """Ngưỡng hovering: ưu tiên đề xuất TỪ PHÂN BỐ (p0b_hover_threshold.py, khối ALL); không có → giá trị tạm của p0_egomotion."""
    p = P0 / "egomotion_seq.parquet"
    if not p.exists():
        return None, None
    s = pd.read_parquet(p)
    thr = json.loads((P0 / "egomotion_summary.json").read_text())["hover_threshold_px"]
    ht = P0 / "hover_threshold.json"
    if ht.exists():
        h = json.loads(ht.read_text(encoding="utf-8"))
        prop = h.get("by_dataset", {}).get("ALL", {}).get("proposed_threshold_px") if h.get("status") == "OK" else None
        if prop:
            thr = prop
    return dict(zip(s.seq, np.where(s.shift_px_median < thr, "hovering", "moving"))), thr


def censor_flags(d, last_by_seq, first_by_seq):
    """Bảo đảm có seq_last / right_censored / left_censored (E1 tính từ start, D)."""
    if "right_censored" not in d.columns:
        d = d.copy()
        d["seq_last"] = d.seq.map(last_by_seq).astype(int)
        d["seq_first"] = d.seq.map(first_by_seq).astype(int)
        d["right_censored"] = d.start + d.D - 1 >= d.seq_last
        d["left_censored"] = d.start <= d.seq_first
    return d


def load_events(tag, fr):
    last, first = fr.groupby("seq").frame.max(), fr.groupby("seq").frame.min()
    ev = {}
    for e in ("E1", "E2", "E3"):
        p = P0 / f"events_{e}_{tag}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if e == "E3":
                d = d[~d.never_visible]
            ev[e] = censor_flags(d, last, first)
    return ev


def birth_window(d, W=W_BIRTH):
    """Cửa sổ sinh: start ≤ seq_last − W, bỏ kiểm duyệt trái."""
    return d[(~d.left_censored) & (d.start <= d.seq_last - W)]


# ---------------------------------------------------------------- Kaplan–Meier có trọng số (bootstrap vector hoá)
class KM:
    """KM cho D với cờ complete (= không kiểm duyệt phải). mass(W): W (B × n_seq) trọng số chuỗi."""

    def __init__(self, D, complete, seq_codes, n_seq):
        D = np.asarray(D, dtype=np.int64)
        self.T = np.unique(D)
        j = np.searchsorted(self.T, D)
        self.A_all = np.zeros((n_seq, len(self.T)))
        self.A_cmp = np.zeros((n_seq, len(self.T)))
        np.add.at(self.A_all, (seq_codes, j), 1.0)
        np.add.at(self.A_cmp, (seq_codes, j), np.asarray(complete, dtype=float))

    def mass(self, W):
        W = np.atleast_2d(W)
        d = W @ self.A_cmp
        n_j = np.cumsum((W @ self.A_all)[:, ::-1], axis=1)[:, ::-1]  # số ở nguy cơ tại T_j (D ≥ T_j)
        h = np.divide(d, n_j, out=np.zeros_like(d), where=n_j > 0)
        Sv = np.cumprod(1.0 - h, axis=1)
        prev = np.hstack([np.ones((Sv.shape[0], 1)), Sv[:, :-1]])
        return prev - Sv, Sv[:, -1]  # khối lượng tại T_j, đuôi (gán tại T_max)

    def expect(self, g, W):
        """E[g(D)] theo KM; g: hàm vector trên T. Đuôi gán g(T_max) (cận trên nếu g giảm)."""
        m, tail = self.mass(W)
        gv = np.asarray(g(self.T), dtype=float)
        return m @ gv + tail * gv[-1]

    def quantile(self, p):
        m, _ = self.mass(np.ones((1, self.A_all.shape[0])))
        F = np.cumsum(m[0])
        i = np.searchsorted(F, p - 1e-12)
        return (float(self.T[i]), True) if i < len(self.T) else (float(self.T[-1]), False)


def km_for(d, seqs):
    """KM trên mọi sự kiện không kiểm duyệt trái."""
    k = d[~d.left_censored]
    codes = pd.Categorical(k.seq, categories=seqs).codes
    return KM(k.D.values, ~k.right_censored.values, codes, len(seqs)), k


def km_dq(km, fps):
    out = {}
    for p in (10, 50, 90):
        v, ok = km.quantile(p / 100)
        out[f"p{p}"] = v
        out[f"p{p}_reached"] = ok
        if fps:
            out[f"p{p}_s"] = v / fps
    m, tail = km.mass(np.ones((1, km.A_all.shape[0])))
    out["share_D_lt_8"] = float(m[0][km.T < 8].sum())
    out["tail_mass"] = float(tail[0])
    return out


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}
    stats["meta"] = dict(K_GRID=K_GRID, R_GRID=R_GRID, R_RECALL=R_RECALL, CI_K=CI_K, CI_R=CI_R, B=B, seed=SEED, W_birth=W_BIRTH,
                         windows=list(WINDOWS), main_window="birth_W120",
                         formula="N=floor(D/S)+Bernoulli(frac(D/S)), S=k*R", ci="percentile 95%, bootstrap theo chuỗi")
    stats["data"] = {}
    stats["E0"] = {}
    stats["D_quantiles"] = {}
    stats["D_quantiles_km"] = {}
    stats["censoring"] = {}
    stats["E3_arrival"] = {}
    grid, strata = [], []
    hov, thr = hovering_map()
    for tag, ds in DATASETS.items():
        pj = P0 / f"parse_{tag}.json"
        if not pj.exists() or json.loads(pj.read_text(encoding="utf-8")).get("status") != "OK":
            stats["data"][ds] = {"status": "NO_DATA"}
            continue
        info = json.loads(pj.read_text(encoding="utf-8"))
        stats["data"][ds] = {k: v for k, v in info.items() if k != "seq_meta"}
        stats["data"][ds]["n_seq_by_split"] = info["n_seq_by_split"]
        e0 = pd.read_parquet(P0 / f"events_E0_{tag}.parquet")
        fr = pd.read_parquet(P0 / f"frames_{tag}.parquet")
        stats["E0"][ds] = {}
        for vd in ("main", "strict"):
            d = e0[e0.vis_def == vd]
            col = "M" if vd == "main" else "M_strict"
            stats["E0"][ds][vd] = dict(
                rho_ev_median=float(d.rho_ev.median()), rho_ev_min=float(d.rho_ev.min()), rho_ev_max=float(d.rho_ev.max()),
                rho_ev_pooled=float((fr[col] > 0).mean()), n_seq_rho_lt_0_9=int((d.rho_ev < 0.9).sum()),
                M_median=float(fr[col].median()), M_mean=float(fr[col].mean()),
                M_p10=float(fr[col].quantile(0.1)), M_p90=float(fr[col].quantile(0.9)), M_max=int(fr[col].max()),
                share_frames_M_1_5=float(fr[col].between(1, 5).mean()), share_frames_M_6_20=float(fr[col].between(6, 20).mean()),
                share_frames_M_gt_20=float((fr[col] > 20).mean()), share_frames_M_0=float((fr[col] == 0).mean()))
        arr = pd.read_parquet(P0 / f"events_E3_arrival_{tag}.parquet")
        stats["E3_arrival"][ds] = {vd: dict(median_per_100f=float(a.arrival_per_100f.median()),
                                            pooled_per_100f=float(100 * a.n_E3.sum() / len(fr)),
                                            n_seq_with_E3=int(len(a)))
                                   for vd, a in arr.groupby("vis_def")}
        ev = load_events(tag, fr)
        stats["D_quantiles"][ds] = {}
        stats["D_quantiles_km"][ds] = {}
        stats["censoring"][ds] = {}
        seqs = sorted(fr.seq.unique())
        W = boot_weights(seqs)  # cùng trọng số cho mọi ô của dataset → so sánh cặp
        Wk = np.vstack([np.ones(len(seqs)), W])  # hàng 0 = ước lượng điểm
        for e, d_all in ev.items():
            for vd in sorted(d_all.vis_def.unique()):
                for grp in sorted(d_all.group.unique()):
                    dd_all = d_all[(d_all.vis_def == vd) & (d_all.group == grp)]
                    if dd_all.empty:
                        continue
                    key = f"{e}|{vd}|{grp}"
                    dd_b = birth_window(dd_all)
                    km, dd_km = km_for(dd_all, seqs)
                    stats["censoring"][ds][key] = dict(
                        n_all=int(len(dd_all)), n_right_censored=int(dd_all.right_censored.sum()),
                        share_right_censored=float(dd_all.right_censored.mean()), n_left_censored=int(dd_all.left_censored.sum()),
                        n_birth_window=int(len(dd_b)), n_birth_window_right_censored=int(dd_b.right_censored.sum()),
                        birth_censored_min_D_obs=int(dd_b[dd_b.right_censored].D.min()) if dd_b.right_censored.any() else None,
                        n_km=int(len(dd_km)))
                    # phân vị: v1 (mọi sự kiện), cửa sổ sinh (đúng khi ≤ W), KM
                    stats["D_quantiles"][ds][key] = dq(dd_all.D.values, FPS[ds])
                    qb = dq(dd_b.D.values, FPS[ds])
                    qb.update({f"p{p}_exact": bool(qb[f"p{p}"] <= W_BIRTH) for p in (10, 50, 90)})
                    stats["D_quantiles"][ds][f"{key}|birth"] = qb
                    stats["D_quantiles_km"][ds][key] = km_dq(km, FPS[ds])
                    if e == "E3":
                        for t3 in sorted(dd_all.e3_type.unique()):
                            stats["D_quantiles"][ds][f"{key}|{t3}"] = dq(dd_all[dd_all.e3_type == t3].D.values, FPS[ds])
                            q3 = dq(dd_b[dd_b.e3_type == t3].D.values, FPS[ds])
                            q3.update({f"p{p}_exact": bool(q3[f"p{p}"] <= W_BIRTH) for p in (10, 50, 90)})
                            stats["D_quantiles"][ds][f"{key}|{t3}|birth"] = q3
                            km3, _ = km_for(dd_all[dd_all.e3_type == t3], seqs)
                            stats["D_quantiles_km"][ds][f"{key}|{t3}"] = km_dq(km3, FPS[ds])
                        stats["D_quantiles"][ds][f"{key}|M_at_birth"] = dict(
                            median=float(dd_all.M_at_birth.median()), p10=float(dd_all.M_at_birth.quantile(.1)), p90=float(dd_all.M_at_birth.quantile(.9)))
                        for lab, dsub in (("all", dd_all), ("birth", dd_b)):
                            short = dsub[dsub.D < 8]
                            stats["censoring"][ds][key][f"{lab}_D_lt_8_n"] = int(len(short))
                            stats["censoring"][ds][key][f"{lab}_D_lt_8_share_border"] = float((short.e3_type == "border").mean()) if len(short) else None
                    for k in K_GRID:
                        for R in R_GRID:
                            S = k * R
                            ci_cell = k in CI_K and R in CI_R
                            for win, dd in (("birth_W120", dd_b), ("all_v1", dd_all)):
                                m = {n: float(v.mean()) for n, v in metrics(dd.D.values, S).items()}
                                rec = dict(dataset=ds, event=e, vis_def=vd, group=grp, window=win, k=k, R=R, S=S,
                                           exact_r1=bool(win == "birth_W120" and S <= W_BIRTH),
                                           n_events=len(dd), n_seq=int(dd.seq.nunique()), **m)
                                if ci_cell:
                                    for n, (lo, hi) in boot(dd, S, seqs, W).items():
                                        rec[f"{n}_lo"], rec[f"{n}_hi"] = lo, hi
                                grid.append(rec)
                            rec = dict(dataset=ds, event=e, vis_def=vd, group=grp, window="km", k=k, R=R, S=S, exact_r1=False,
                                       n_events=len(dd_km), n_seq=int(dd_km.seq.nunique()))
                            vals = {"share_short": km.expect(lambda T: p_zero(T, S), Wk)}
                            for r in R_RECALL:
                                vals[f"miss_r{r}"] = km.expect(lambda T, r=r: miss_r(T, S, r), Wk)
                            for n, v in vals.items():
                                rec[n] = float(v[0])
                                if ci_cell:
                                    rec[f"{n}_lo"], rec[f"{n}_hi"] = float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))
                            grid.append(rec)
                    # phân tầng (chỉ VEHICLE, ô CI) — trên cửa sổ sinh
                    if grp != "VEHICLE":
                        continue
                    sd = dd_b.copy()
                    sd["size_bin"] = size_bin(sd.sqrt_area_median)
                    sd["speed_tertile"] = pd.qcut(sd.speed_px_median.rank(method="first"), 3, labels=["slow", "mid", "fast"]).astype(str) \
                        if sd.speed_px_median.notna().sum() >= 3 else "na"
                    sd["M_bin"] = M_bin(sd.M_at_start)
                    svars = ["size_bin", "speed_tertile", "M_bin"]
                    if "altitude" in sd.columns:
                        svars.append("altitude")
                    if e == "E3":
                        svars.append("e3_type")
                    if hov is not None:
                        sd["ego"] = sd.seq.map(hov).fillna("na")
                        svars.append("ego")
                    for sv in svars:
                        for lvl, g in sd.groupby(sv):
                            for k in CI_K:
                                for R in CI_R:
                                    S = k * R
                                    m = {n: float(v.mean()) for n, v in metrics(g.D.values, S).items()}
                                    strata.append(dict(dataset=ds, event=e, vis_def=vd, window="birth_W120", stratum_var=sv, stratum=str(lvl),
                                                       k=k, R=R, S=S, n_events=len(g), n_seq=int(g.seq.nunique()), D_p50=float(g.D.median()),
                                                       D_p50_exact=bool(g.D.median() <= W_BIRTH),
                                                       speed_px_p50=float(g.speed_px_median.median()), **m))
                    if e == "E2":
                        tert = sd.groupby("speed_tertile").speed_px_median.agg(["min", "max"]).to_dict("index")
                        stats.setdefault("speed_tertile_edges", {})[f"{ds}|{vd}"] = tert
    gdf = pd.DataFrame(grid)
    sdf = pd.DataFrame(strata)
    gdf.to_csv(P0 / "exposure_grid.csv", index=False)
    sdf.to_csv(P0 / "strata.csv", index=False)
    stats["exposure"] = grid
    stats["strata"] = strata
    stats["hover_threshold_px"] = thr

    # quyết định CP0 (E3, VEHICLE, VISIBLE chính) — cửa sổ sinh (chính); v1 để so sánh
    dec = {}
    for ds in gdf.dataset.unique() if len(gdf) else []:
        def cell(k, R, win, e="E3"):
            r = gdf[(gdf.dataset == ds) & (gdf.event == e) & (gdf.vis_def == "main") & (gdf.group == "VEHICLE")
                    & (gdf.window == win) & (gdf.k == k) & (gdf.R == R)]
            return r.iloc[0].to_dict() if len(r) else None
        c5, c12 = cell(5, 10, "birth_W120"), cell(12, 10, "birth_W120")
        o5, o12 = cell(5, 10, "all_v1"), cell(12, 10, "all_v1")
        dec[ds] = dict(window="birth_W120",
                       share_short_E3_k5_R10=c5["share_short"], share_short_E3_k5_R10_ci=[c5["share_short_lo"], c5["share_short_hi"]],
                       share_short_E3_k12_R10=c12["share_short"], share_short_E3_k12_R10_ci=[c12["share_short_lo"], c12["share_short_hi"]],
                       B_trigger_lt_5pct=bool(c5["share_short"] < 0.05), C_part1_lt_1pct=bool(c12["share_short"] < 0.01),
                       v1_share_short_E3_k5_R10=o5["share_short"], v1_share_short_E3_k12_R10=o12["share_short"])
    stats["decision"] = dec
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("wrote", STATS, "grid", len(gdf), "strata", len(sdf))
    print(json.dumps(dec, indent=1))
    print(json.dumps(stats["censoring"], indent=1))


if __name__ == "__main__":
    main()
