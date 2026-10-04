"""P2A-C1: sinh PREREG_28.md + prereg_cells.json TỪ SỐ ĐO (không hash/tag — đóng băng là quyết định của anh Đạt).

Đầu vào
  results/p2/channel_train.json  (p2_estimate_channel.py; PHẢI có split == "train" — L8)
      { "split": "train", "simulated": bool, "dataset": "UAVDT", "lambda_per_frame": float, "rho_onset": {"1":..,"3":..,"5":..},
        "operating_points": [ {"cue": str, "theta": float|null, "w": int, "q_in": float, "q_out": float,
                               "q_b": {"all"|"hovering"|"moving": float}, "q_o": float|null, "c": float} , ... ],
        "r_levels": {"<detector level>": r, ...} }
  results/p2/bench_openvino.json (tuỳ chọn; chỉ để ghi nguồn c nếu operating point không có c)
  results/p0_stats.json          (F_D: dùng sự kiện E3 TRAIN cửa sổ sinh từ results/p0/events_E3_uavdt.parquet)
Dự đoán (THEORY_28 §5–§6). Khung sau cửa sổ riêng: mặc định MEAN-FIELD (q_bg = ρ q_in + (1−ρ) q_out, vì A7 cho thấy
  A-sparse quá bảo thủ với cửa sổ KPI dài); tuỳ chọn A-sparse (--mode asparse):
  matched cost: Δ_pred = mean_e[ miss_r(T_e, 1/(a_G + c)) − miss_G(T_e; w, R, q_in, q_out,eff(M), r) ]
      q_out,eff(M) = 1 − (1 − q_b)(1 − q_o)^M tại M đại diện của bin (Lemma P3.1); không có q_o → q_out đo trực tiếp
      dấu dự đoán "+" nếu Δ_pred > MARGIN, ngược lại "≤0"
  iso-KPI (latency): H2 — gate khởi phát rẻ hơn iff ρ1 + (1−ρ1) q_out,eff + c < (1−ε)/T   (r=1; r<1 ghi "numeric")
Ô xác nhận: ≥ 30 sự kiện và ≥ 3 chuỗi (đếm trên TRAIN làm đại diện; kiểm lại trên TEST khi chấm).
Luật P2C (28/09, chốt TRƯỚC khi có kênh thật):
  B1 θ*: mỗi cue MỘT ngưỡng, chọn trên TRAIN: θ* = argmax_θ [q_in(θ) − q_out(θ)] ở w = 5 (Youden kênh khởi phát),
     ràng buộc a_G(θ*) = a_gate(ρ5, q_in, q_out, R = 30) + c ≤ 0,25; hoà → a_G nhỏ hơn  (theta_star()).
  B2 họ xác nhận: detector chính (main_level, mặc định mức r cao nhất) cho matched cost + iso-KPI; H8 = cặp (chính, mức khác);
     R = 30 (gate ∨ refresh — chính sách chính KE_HOACH); R = ∞ chỉ mô tả; L_max ∈ {3,5,10}; ε ∈ {2 %, 5 %};
     M_bin × ego; ô cần ≥ 30 sự kiện, ≥ 3 chuỗi. Nếu > FAMILY_MAX (80) thì thu gọn theo REDUCTION_ORDER (thứ tự cố định,
     chỉ dựa trên số ô, không dựa trên kết quả) cho tới khi ≤ 80.
Chạy: python code/p2/p2_prereg_build.py [--channel PATH] [--out-md PATH] [--out-json PATH]
"""
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from theory_checks import a_gate, lemma_miss, miss_gate  # noqa: E402

P0 = ROOT / "results" / "p0"
MARGIN = 0.005
M_REP = {"1-5": 3, "6-20": 13, ">20": 30}
KPIS = ["miss", 3, 5, 10, 30]
CONF_KPIS = {"3", "5", "10"}
ISO_EPS = [0.01, 0.02, 0.05]
CONF_EPS = {0.02, 0.05}
R_SET = [math.inf, 30]
W_BIRTH = 120
CONF_R = 30          # R của họ xác nhận (gate ∨ refresh); R = ∞ chỉ mô tả
W_STAR = 5           # cửa sổ khởi phát dùng chọn θ* (B1)
A_MAX = 0.25         # ràng buộc a_G(θ*) ≤ 0,25 (B1)
FAMILY_MAX = 80
M_BINS = ["1-5", "6-20", ">20"]
CUE_C_MAX = 1.0      # P2F luật chi phí (trước freeze): cue có c ≥ 1 (đắt bằng/hơn detector) bị loại — Thm G3: gate không thể thắng
H8_DELTA = 0.05      # P2F-Q2: mức recall thấp hợp lệ khi r_main − r_low ≥ 0,05 (score 0,25, TRAIN, ước lượng điểm, IoU 0,5)
H8_CANDIDATES = ["yolo26n_1024", "yolo26s_640", "yolo26n_640"]
H8_MAX_PAIRS = 2
REDUCTION_ORDER = [  # P2C-B2: áp lần lượt khi họ > FAMILY_MAX; 2 bước đầu theo PROMPT P2C, các bước sau Claude đề xuất (chờ duyệt)
    ("drop_eps5", "bỏ ε = 5 % (giữ 2 %)"),
    ("drop_L10", "bỏ L_max = 10"),
    ("H8_pool_M", "H8 gộp M_bin (H8 nói về r, không về mật độ)"),
    ("H8_L3", "H8 chỉ L_max = 3"),
    ("H8_pool_ego", "H8 gộp ego"),
    ("iso_L3", "iso-KPI chỉ L_max = 3"),
    ("matched_L3", "matched cost chỉ L_max = 3"),
]


def cell_id(dataset, m, kpi, M_bin, ego, rlevel, cue, w, R, theta=None):
    t = "" if theta is None else f"|th{theta:g}"
    return f"{dataset}|m{m:g}|{kpi}|M{M_bin}|{ego}|{rlevel}|{cue}{t}|w{w}|R{'inf' if R == math.inf else int(R)}"


def train_events(dataset="uavdt"):
    fr = pd.read_parquet(P0 / f"frames_{dataset}.parquet")
    last = fr.groupby("seq").frame.max()
    e3 = pd.read_parquet(P0 / f"events_E3_{dataset}.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main") & (e3.split == "train")]
    e3 = e3[e3.start <= e3.seq.map(last) - W_BIRTH]
    e3["M_bin"] = pd.cut(e3.M_at_birth, [-np.inf, 5.5, 20.5, np.inf], labels=["1-5", "6-20", ">20"]).astype(str)
    ap = P0 / "audit_prebirth_train.json"  # P2G: chỉ cho phụ lục độ nhạy onset_alt (KHÔNG lọc sự kiện)
    if ap.exists():
        a = pd.DataFrame(json.loads(ap.read_text(encoding="utf-8"))["events"])[["seq", "track_id", "shift"]]
        e3 = e3.merge(a, on=["seq", "track_id"], how="left")
        e3["D_alt"] = e3.D + e3["shift"].fillna(0).astype(int)
    return e3


MODE = "meanfield"  # "meanfield" (mặc định) | "asparse" (bảo thủ cho gate)


def predict_delta(D, kpi, w, R, q_in, q_out, c, rho, r, mode=None):
    """Khung sau cửa sổ khởi phát riêng: A-sparse → q_out; mean-field → nằm trong cửa sổ xe khác với xác suất ρ,
    tức q_bg = ρ q_in + (1 − ρ) q_out (bỏ qua tương quan giữa các khung)."""
    mode = mode or MODE
    T = D if kpi == "miss" else np.minimum(D, kpi + 1)
    a = min(a_gate(rho, q_in, q_out, R) + c, 1.0)
    mP = np.mean(lemma_miss(T, 1.0 / a, r))
    q_bg = q_out if mode == "asparse" else rho * q_in + (1 - rho) * q_out
    mG = np.mean([miss_gate(int(t), w, R, q_in, q_bg, r) for t in T])
    return float(mP - mG), float(a), float(mP), float(mG)


def r_dagger(D, kpi, w, R, q_in, q_out, c, rho, mode=None, n_grid=401):
    """P2H-Q7: r† = argmax_r Δ_pred(r) trên [0, 1] cho cấu hình của ô (cùng mô hình predict_delta: T_e = min(D, L+1) của ô,
    a = a_G + c không phụ thuộc r, khung nền mean-field/A-sparse, refresh R). Lưới n_grid điểm + tinh chỉnh bước 1e-4 quanh đỉnh.
    Remark G2′: Δ lõm (R = ∞, T a < 1) ⇒ Δ không tăng trên [r†, 1]; H8 chỉ dự đoán khi cả hai recall > r†."""
    mode = mode or MODE
    T = np.minimum(np.asarray(D), kpi + 1)
    u, cnt = np.unique(T, return_counts=True)
    wts = cnt / cnt.sum()
    a = min(a_gate(rho, q_in, q_out, R) + c, 1.0)
    q_bg = q_out if mode == "asparse" else rho * q_in + (1 - rho) * q_out

    def f(r):
        return float(np.sum(wts * (lemma_miss(u, 1.0 / a, r) - np.array([miss_gate(int(t), w, R, q_in, q_bg, r) for t in u]))))
    g = np.linspace(0, 1, n_grid)
    i = int(np.argmax([f(r) for r in g]))
    fine = np.arange(max(0.0, g[i] - 1 / (n_grid - 1)), min(1.0, g[i] + 1 / (n_grid - 1)) + 1e-12, 1e-4)
    return float(fine[int(np.argmax([f(r) for r in fine]))])


def s_star_bisect(T, r, eps, S_max=400.0, iters=40):
    """S thực lớn nhất với mean Lemma1(T, S, r) ≤ eps (giả định đơn điệu); None nếu S=1 cũng không đạt."""
    f = lambda S: float(np.mean(lemma_miss(T, S, r)))  # noqa: E731
    if f(1.0) > eps + 1e-12:
        return None
    if f(S_max) <= eps:
        return S_max
    lo, hi = 1.0, S_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) <= eps + 1e-12 else (lo, mid)
    return lo


def theta_star(rows, rho5, R=CONF_R, a_max=A_MAX):
    """B1: rows = điểm lưới (theta, q_in, q_out, c) của MỘT cue ở w = W_STAR. θ* = argmax (q_in − q_out) với
    a_G = a_gate(ρ5, q_in, q_out, R) + c ≤ a_max; hoà (tới 1e-12) → a_G nhỏ hơn. None nếu không điểm nào thoả ràng buộc."""
    best = None
    for r in rows:
        a = a_gate(rho5, r["q_in"], r["q_out"], R) + (r.get("c") or 0.0)
        if a > a_max + 1e-12:
            continue
        key = (round(r["q_in"] - r["q_out"], 12), -a)
        if best is None or key > best[0]:
            best = (key, r, a)
    return None if best is None else dict(best[1], a_G_star=float(best[2]), J_star=float(best[0][0]))


def select_operating_points(channel):
    """Một operating point / cue theo theta_star trên các operating point w = W_STAR của kênh. Trả (ops, log)."""
    rho5 = channel["rho_onset"][str(W_STAR)]
    by_cue = {}
    for op in channel["operating_points"]:
        if int(op["w"]) == W_STAR:
            by_cue.setdefault(op["cue"], []).append(op)
    ops, log = [], []
    for cue, rows in sorted(by_cue.items()):
        c = max((r.get("c") or 0.0) for r in rows)
        if c >= CUE_C_MAX:  # luật chi phí: loại trước khi chọn θ*
            log.append(dict(cue=cue, n_candidates=len(rows), theta=None, q_in=None, q_out=None, a_G=None, J=None, c=c,
                            excluded=f"c = {c:.2f} ≥ {CUE_C_MAX:g} (luật chi phí, Thm G3)", J_unconstrained=None, binding=None))
            continue
        s = theta_star(rows, rho5)
        free = max(rows, key=lambda r: r["q_in"] - r["q_out"])  # Youden không ràng buộc a_G
        log.append(dict(cue=cue, n_candidates=len(rows), theta=None if s is None else s.get("theta"),
                        q_in=None if s is None else s["q_in"], q_out=None if s is None else s["q_out"],
                        a_G=None if s is None else s["a_G_star"], J=None if s is None else s["J_star"], c=c, excluded=None,
                        J_unconstrained=float(free["q_in"] - free["q_out"]),
                        binding=bool(s is None or free["theta"] != s["theta"])))
        if s is not None:
            ops.append(s)
    return ops, log


def h8_rule(r_levels, main, candidates=H8_CANDIDATES, delta=H8_DELTA, max_pairs=H8_MAX_PAIRS):
    """P2F-Q2 (cố định, PREREG §2): mức hợp lệ = ứng viên có r_main − r ≥ delta. Cặp H8 = (main, mức hợp lệ), mức r thấp nhất
    trước, tối đa max_pairs cặp. status: "pairs" | "none_valid" (H8 → mô tả) | "pending" (còn ứng viên chưa có r, chưa mức nào hợp lệ)."""
    r0 = r_levels[main]
    gaps = {lv: float(r0 - r_levels[lv]) for lv in candidates if lv in r_levels}
    valid = sorted((lv for lv, g in gaps.items() if g >= delta - 1e-12), key=lambda lv: r_levels[lv])
    missing = [lv for lv in candidates if lv not in r_levels]
    pairs = [(main, lv) for lv in valid[:max_pairs]]
    status = "pairs" if pairs else ("pending" if missing else "none_valid")
    return dict(delta=delta, main=main, r_main=float(r0), gaps=gaps, valid=valid, missing=missing, pairs=[list(p) for p in pairs],
                status=status, max_pairs=max_pairs)


def _levels(channel):
    r_items = sorted(channel["r_levels"].items(), key=lambda kv: -kv[1])
    main = channel.get("main_level") or r_items[0][0]
    if "h8_pairs" in channel:  # cặp đặt tay (chỉ dùng cho kênh mô phỏng / test)
        return r_items, main, [tuple(pr) for pr in channel["h8_pairs"]], None
    rule = h8_rule(channel["r_levels"], main)
    return r_items, main, [tuple(pr) for pr in rule["pairs"]], rule


def family_confirmatory(c, active):
    """Ô có thuộc họ xác nhận dưới tập bước thu gọn `active` không."""
    if not c.get("eligible"):
        return False
    kind, L = c["comparison"], c.get("L")
    if "drop_eps5" in active and kind == "iso_kpi" and c.get("eps") == 0.05:
        return False
    if "drop_L10" in active and L == 10:
        return False
    if kind == "r_pair":
        want = "M+ego" if "H8_pool_ego" in active else "M" if "H8_pool_M" in active else None
        if c.get("pool") != want or ("H8_L3" in active and L != 3):
            return False
    elif c.get("pool") is not None:
        return False
    if "iso_L3" in active and kind == "iso_kpi" and L != 3:
        return False
    if "matched_L3" in active and kind == "matched_cost" and L != 3:
        return False
    return True


def apply_family_rules(cells, fmax=FAMILY_MAX, single_ego=False):
    """Áp REDUCTION_ORDER cho tới khi họ ≤ fmax; ghi cờ `confirmatory`. Trả log từng bước."""
    active = []
    log = [dict(step="start", n=sum(family_confirmatory(c, active) for c in cells))]
    for step, desc in REDUCTION_ORDER:
        if log[-1]["n"] <= fmax:
            break
        if step == "H8_pool_ego" and single_ego:
            continue
        active.append(step)
        log.append(dict(step=step, desc=desc, n=sum(family_confirmatory(c, active) for c in cells)))
    for c in cells:
        c["confirmatory"] = family_confirmatory(c, active)
    # P2H-Q7: SAU thu gọn — ô H8 có r_main hoặc r_low ≤ r† (dự đoán "—") ra khỏi họ xác nhận; không kích hoạt lại thu gọn
    drop = [c for c in cells if c["confirmatory"] and c.get("rdag_ok") is False]
    for c in drop:
        c["confirmatory"] = False
    log.append(dict(step="H8_rdagger", desc="bỏ ô H8 có r_main hoặc r_low ≤ r† (Q7)", n=log[-1]["n"] - len(drop)))
    return dict(fmax=fmax, active=active, log=log, n_final=log[-1]["n"], over_limit=bool(log[-1]["n"] > fmax))


def is_confirmatory(kpi, n_ev, n_seq, eps=None):
    """Điều kiện cơ sở (27/09): KPI latency Lmax ∈ {3,5,10}, ε ∈ {2%,5%} cho iso-KPI, ≥ 30 sự kiện, ≥ 3 chuỗi.
    Cờ cuối `confirmatory` do apply_family_rules quyết định (thêm R, mức detector, thu gọn)."""
    ok_kpi = str(kpi) in CONF_KPIS and (eps is None or eps in CONF_EPS)
    return bool(ok_kpi and n_ev >= 30 and n_seq >= 3)


def build(channel, e3, dataset_label="UAVDT", m_list=(0.0,), ego_by_seq=None, select=True, fmax=FAMILY_MAX, info=None):
    """Sinh mọi ô (xác nhận + mô tả). select=True: một θ*/cue theo B1. ego_by_seq: {seq: ego} để đếm sự kiện theo ego
    (không có → mọi lớp ego dùng toàn bộ sự kiện, ghi ego_events_split=False). info (dict, tuỳ chọn) nhận log θ*/thu gọn."""
    assert channel.get("split") == "train", "L8: kênh phải ước lượng trên TRAIN"
    rho = {int(k): v for k, v in channel["rho_onset"].items()}
    ops, sel_log = select_operating_points(channel) if select else (channel["operating_points"], [])
    r_items, main, pairs, h8 = _levels(channel)
    rmap = dict(r_items)
    pair_set = set(pairs)
    egos = sorted({k for op in ops for k in op["q_b"]})
    if len(egos) > 1 and "all" in egos:
        egos.remove("all")
    ego_by_seq = ego_by_seq or {}
    e3 = e3.copy()
    e3["ego"] = e3.seq.map(ego_by_seq).fillna("unknown") if ego_by_seq else "all"
    strata = [(eg, mb, None) for eg in egos for mb in M_BINS] + [(eg, "all", "M") for eg in egos]
    if len(egos) > 1:
        strata.append(("all", "all", "M+ego"))
    h8_levels = [(lv, rmap[lv]) for lv, _ in r_items if any(lv in pr for pr in pairs)]
    cells, rdag = [], {}
    for op in ops:
        w = int(op["w"])
        rho_w = rho.get(w)
        c = op.get("c", 0.0)
        for ego, mb, pool in strata:
            sub = e3 if (ego == "all" or not ego_by_seq) else e3[e3.ego == ego]
            if mb != "all":
                sub = sub[sub.M_bin == mb]
            D = sub.D.values
            n_ev, n_seq = int(len(D)), int(sub.seq.nunique())
            if n_ev == 0:
                continue
            q_b = op["q_b"].get(ego, op["q_b"].get("all", float(np.mean(list(op["q_b"].values())))))
            Mrep = M_REP[mb] if mb != "all" else (float(np.median(sub.M_at_birth)) if "M_at_birth" in sub else 13.0)
            q_out = op["q_out"] if op.get("q_o") is None else 1 - (1 - q_b) * (1 - op["q_o"]) ** Mrep
            enough = bool(n_ev >= 30 and n_seq >= 3)
            for m in m_list:
                for R in ([CONF_R] if pool else R_SET):
                    Rs = "inf" if R == math.inf else int(R)
                    base = dict(dataset=dataset_label, m=m, M_bin=mb, ego=ego, pool=pool, cue=op["cue"], theta=op.get("theta"), w=w, R=Rs,
                                q_in=op["q_in"], q_out_eff=q_out, c=c, n_events_train=n_ev, n_seq_train=n_seq, mode=MODE,
                                ego_events_split=bool(ego_by_seq))
                    dl = {}
                    for rlevel, r in (h8_levels if pool else r_items):
                        for kpi in (KPIS if not pool else (3, 5, 10)):
                            dlt, a, mP, mG = predict_delta(D, kpi, w, R, op["q_in"], q_out, c, rho_w, r)
                            dl[(rlevel, kpi)] = dlt
                            if pool:
                                continue
                            alt = {}
                            if "D_alt" in sub and str(kpi) in CONF_KPIS and R == CONF_R and rlevel == main:
                                # phụ lục (P2G): onset_alt = khung sớm nhất b−5…b−1 có detector khớp → cửa sổ KPI dài thêm `shift`
                                da, _, mPa, mGa = predict_delta(sub.D_alt.values, kpi, w, R, op["q_in"], q_out, c, rho_w, r)
                                alt = dict(delta_pred_onset_alt=da, miss_P_pred_onset_alt=mPa, miss_G_pred_onset_alt=mGa)
                            cells.append(dict(base, **alt, id=cell_id(dataset_label, m, kpi, mb, ego, rlevel, op["cue"], w, R, op.get("theta")),
                                              kpi=str(kpi), L=(kpi if kpi != "miss" else None), r_level=rlevel, r=r, a_cost=a,
                                              miss_P_pred=mP, miss_G_pred=mG, delta_pred=dlt,
                                              sign_pred=("+" if dlt > MARGIN else "<=0"), comparison="matched_cost",
                                              eligible=bool(is_confirmatory(kpi, n_ev, n_seq) and R == CONF_R and rlevel == main),
                                              hypothesis=("H1" if kpi == "miss" else "H3/H5" if str(kpi) in CONF_KPIS else "H3-desc")))
                        if pool:
                            continue
                        # iso-KPI (H2/H6): gate khả thi (miss_G ≤ ε) và rẻ hơn periodic ở cùng ε
                        for Lm in (3, 5, 10, 30):
                            T = np.minimum(D, Lm + 1)
                            a_G = min(a_gate(rho_w, op["q_in"], q_out, R) + c, 1.0)
                            q_bg = q_out if MODE == "asparse" else rho_w * op["q_in"] + (1 - rho_w) * q_out
                            mG = float(np.mean([miss_gate(int(t), w, R, op["q_in"], q_bg, r) for t in T]))
                            for eps in ISO_EPS:
                                S = s_star_bisect(T, r, eps)
                                a_P = (1.0 / S) if S else math.inf
                                cheaper = bool(mG <= eps and a_G < a_P)
                                cells.append(dict(base, id=cell_id(dataset_label, m, f"iso_L{Lm}_eps{eps:g}", mb, ego, rlevel, op["cue"], w, R,
                                                                   op.get("theta")),
                                                  kpi=f"iso_L{Lm}", L=Lm, eps=eps, r_level=rlevel, r=r, a_cost=a_G, a_P_iso=a_P, miss_G_pred=mG,
                                                  sign_pred=("+" if cheaper else "<=0"), comparison="iso_kpi",
                                                  eligible=bool(is_confirmatory(Lm, n_ev, n_seq, eps) and R == CONF_R and rlevel == main),
                                                  hypothesis=("H6" if Lm == 30 else "H2" if eps in CONF_EPS else "H2-eps1-appendix")))
                    # H8: Δ(latency) tăng khi r giảm (Remark G2′) — cặp mức detector (mức r cao trước)
                    lv = h8_levels if pool else r_items
                    for (hi_l, hi_r), (lo_l, lo_r) in [(lv[i], lv[j]) for i in range(len(lv)) for j in range(i + 1, len(lv))]:
                        in_fam = (hi_l, lo_l) in pair_set or (lo_l, hi_l) in pair_set
                        if pool and not in_fam:
                            continue
                        for kpi in (3, 5, 10):
                            d = dl[(lo_l, kpi)] - dl[(hi_l, kpi)]
                            rd = None
                            if in_fam and R == CONF_R:  # P2H-Q7: r† của cấu hình (cue, θ*, tầng, L) — điều kiện của Remark G2′
                                key = (op["cue"], ego, mb, pool, kpi)
                                if key not in rdag:
                                    rdag[key] = r_dagger(D, kpi, w, R, op["q_in"], q_out, c, rho_w)
                                rd = rdag[key]
                            ok = None if rd is None else bool(hi_r > rd and lo_r > rd)
                            cells.append(dict(base, id=cell_id(dataset_label, m, f"H8_L{kpi}", mb, ego, f"{hi_l}>{lo_l}", op["cue"], w, R,
                                                               op.get("theta")),
                                              kpi=f"H8_L{kpi}", L=kpi, r_level=f"{hi_l}>{lo_l}", r_hi=hi_r, r_lo=lo_r, ddelta_pred=d,
                                              r_dagger=rd, rdag_ok=ok,
                                              sign_pred=("—" if ok is False else "+" if d > MARGIN else "<=0"), comparison="r_pair",
                                              eligible=bool(enough and in_fam and R == CONF_R), hypothesis="H8"))
    fam = apply_family_rules(cells, fmax, single_ego=len(egos) <= 1)
    if info is not None:
        info.update(theta_star=sel_log, family=fam, main_level=main, h8_pairs=[list(pr) for pr in pairs], h8_rule=h8, egos=egos,
                    ego_events_split=bool(ego_by_seq), conf_R=CONF_R, w_star=W_STAR, a_max=A_MAX)
    return cells


def write_md(cells, channel, path, info=None):
    info = info or {}
    df = pd.DataFrame(cells)
    conf = df[df.confirmatory]
    L = ["# PREREG_28 — sinh tự động (CHƯA đóng băng)", "",
         f"*Sinh bởi `code/p2/p2_prereg_build.py` — {dt.datetime.now():%Y-%m-%d %H:%M}. Kênh: split={channel['split']}, "
         f"simulated={channel.get('simulated')}. KHÔNG hash/tag: đóng băng (sha256 + git tag) là quyết định của anh Đạt, sau khi kênh đo thật trên TRAIN.*", "",
         f"Công thức: THEORY_28 §5–§6, chế độ khung nền = {MODE}. Luật họ + θ* + tiêu chí bác bỏ: PREREG_28_DRAFT.md §2, §4 "
         "(quyết định 01-10-2026: PREREG_28_DRAFT.md §Quyết định).", ""]
    if not channel.get("simulated"):
        L += real_sections(channel, info, df)
    if info.get("theta_star"):
        L += [f"## θ* (B1: argmax q_in − q_out ở w = {info['w_star']}, a_G(R = {info['conf_R']}) ≤ {info['a_max']}; hoà → a_G nhỏ hơn)", "",
              "| cue | số θ ứng viên | θ* | q_in | q_out | J | a_G | c | J không ràng buộc | ràng buộc a_G chặn | loại |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda x, d=4: "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))  # noqa: E731
        for s in info["theta_star"]:
            bd = "—" if s.get("binding") is None else ("có" if s["binding"] else "không")
            L.append(f"| {s['cue']} | {s['n_candidates']} | {f(s['theta'])} | {f(s['q_in'])} | {f(s['q_out'])} | {f(s['J'])} | {f(s['a_G'])} | "
                     f"{f(s.get('c'), 3)} | {f(s.get('J_unconstrained'))} | {bd} | {s.get('excluded') or '—'} |")
        L.append("")
    if info.get("family"):
        fam = info["family"]
        L += [f"## Họ xác nhận: {fam['n_final']} ô (giới hạn {fam['fmax']}{' — VẪN VƯỢT' if fam['over_limit'] else ''})", "",
              f"Detector chính: {info['main_level']}; cặp H8: {', '.join('>'.join(pr) for pr in info['h8_pairs']) or '— (' + ((info.get('h8_rule') or {}).get('status') or '') + ')'}; R = {info['conf_R']}; "
              f"ego: {', '.join(info['egos'])} (sự kiện tách theo ego: {'có' if info['ego_events_split'] else 'KHÔNG — chưa có nhãn ego'}).", "",
              "| bước | mô tả | số ô sau bước |", "|---|---|---|"]
        for s in fam["log"]:
            L.append(f"| {s['step']} | {s.get('desc', '—')} | {s['n']} |")
        L.append("")
    L += [f"- Tổng ô (gồm mô tả): {len(df)}; ô xác nhận: {len(conf)}; dự đoán gate thắng (+) trong họ: {int((conf.sign_pred == '+').sum())}.",
          f"- Theo giả thuyết (ô xác nhận): " + ", ".join(f"{h}: {int(n)}" for h, n in conf.groupby('hypothesis').size().items()), "",
          "| giả thuyết | KPI | ε | M_bin | ego | detector | cue | dự đoán | Δ dự đoán | n sự kiện / chuỗi (TRAIN) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in conf.sort_values(["hypothesis", "cue", "ego", "M_bin", "kpi"]).itertuples(index=False):
        d = getattr(x, "delta_pred", None)
        d = x.ddelta_pred if x.comparison == "r_pair" else d
        dtxt = "—" if d is None or (isinstance(d, float) and math.isnan(d)) else f"{d:+.4f}"
        eps = getattr(x, "eps", None)
        etxt = "—" if eps is None or (isinstance(eps, float) and math.isnan(eps)) else f"{eps:g}"
        L.append(f"| {x.hypothesis} | {x.kpi} | {etxt} | {x.M_bin} | {x.ego} | {x.r_level} | {x.cue} | {x.sign_pred} | {dtxt} | "
                 f"{x.n_events_train} / {x.n_seq_train} |")
    L += ["", "Danh sách đầy đủ (cả ô mô tả) + Δ dự đoán: prereg_cells.json (cùng thư mục kết quả)."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _j(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f} %"


def real_sections(channel, info, df):
    """§0–§6 của PREREG sinh từ json (P2F). Không số nào gõ tay: mọi số lấy từ channel / audit / hover / parse / recall / bench json."""
    meta = (_j(P0 / "parse_uavdt.json") or {}).get("seq_meta", {})
    au = _j(P0 / "audit_prebirth_train.json")
    hv = (_j(P0 / "hover_threshold.json") or {}).get("by_dataset", {}).get("UAVDT")
    rec = _j(ROOT / "results" / "p2" / "detector_recall_train.json") or {}
    bc = _j(ROOT / "results" / "p2" / "bench_real" / "bench_c_real.json")
    h8 = info.get("h8_rule")
    sm = rec.get("score_min_main")
    L = [f"## §0 Trạng thái kênh: v{channel.get('version', 1)} — r ở score ≥ {sm}; E3: {channel.get('e3_filter')}; "
         f"{channel.get('n_e3_train')} sự kiện E3 TRAIN (VISIBLE chính, trước cắt cửa sổ sinh). "
         "Bản kênh v1 (score 0,05): results/p2/channel_train_score005.json (phụ lục).", ""]
    # §1
    L += ["## §1 Dữ liệu (luật cố định, áp cho cả TEST khi mở)", ""]
    if "M0207" in meta and "M0901" in meta:
        m7, m9 = meta["M0207"], meta["M0901"]
        L += [f"- M0207: {m7['n_img']} ảnh nhưng GT chỉ tới khung {m7['gt_max_frame']} → chỉ dùng khung ≤ {m7['n_frames']} (khung GT cuối); "
              "khung ngoài GT không có nhãn, không phải \"0 xe\".",
              f"- M0901: ảnh thật {m9['W']} × {m9['H']} (GT x+w tối đa {m9['gt_extent'][0]}) → dải biên (border) tính theo bề rộng {m9['W']}; "
              "mọi chuỗi khác dùng kích thước đọc từ ảnh.",
              "- E3 = định nghĩa P0 đầy đủ (VISIBLE: occlusion ≠ 2 và out_of_view ≠ 2), **không loại sự kiện nào**; "
              "audit khung trước onset chỉ mô tả (§5).", ""]
    # §2
    others = [k for k in rec.get("r_levels_by_score", {}) if float(k) != sm]
    L += ["## §2 Luật cố định (trước freeze)", "",
          f"- **Ngưỡng score \"phát hiện\"**: chính = {sm} (mặc định Ultralytics predict, điểm vận hành); {', '.join(others)} = phụ lục mô tả. "
          "Cue tiny_det giữ θ* riêng (ngưỡng trên score của cue, không phải ngưỡng detector)."]
    exc = [s for s in info.get("theta_star", []) if s.get("excluded")]
    L.append(f"- **Luật chi phí**: cue có c ≥ {CUE_C_MAX:g} bị loại trước khi chọn θ* (Thm G3: c ≥ 1 ⇒ gate không thể rẻ hơn tuần hoàn). "
             + ("Đã loại: " + "; ".join(f"{s['cue']} ({s['excluded']})" for s in exc) + "." if exc else "Không cue nào bị loại."))
    L.append(f"- **Cột \"dự đoán\"** (matched cost, H3/H5): ghi \"+\" iff Δ_pred = miss_P − miss_G > δ_min, **δ_min = {MARGIN:g}** (tuyệt đối, "
             "đơn vị tỉ lệ miss); ngược lại \"≤0\". H8: \"+\" iff ΔΔ_pred = Δ_pred(r thấp) − Δ_pred(r chính) > δ_min. "
             "Iso-KPI (H2): \"+\" iff miss_G ≤ ε và a_G < a_P(ε).")
    conf = df[df.confirmatory] if "confirmatory" in df else df.iloc[:0]
    for kind, col, lab in (("matched_cost", "delta_pred", "Δ_pred (H3/H5, họ xác nhận)"),
                           ("r_pair", "ddelta_pred", "ΔΔ_pred (H8, mọi ô R = 30 kể cả mô tả)")):
        sub = conf[conf.comparison == kind] if kind == "matched_cost" else df[(df.comparison == kind) & (df.R.astype(str) == str(CONF_R))]
        if len(sub):
            L.append(f"  - {lab}: lớn nhất {sub[col].max():+.4f}, nhỏ nhất {sub[col].min():+.4f}; số ô \"+\": {int((sub.sign_pred == '+').sum())}/{len(sub)}.")
    if h8:
        L += [f"- **H8 (Q2)**: mức recall thấp hợp lệ khi r_main − r_low ≥ {h8['delta']:g} (score {sm}, TRAIN, ước lượng điểm, IoU {rec.get('iou')}). "
              f"Ứng viên: {', '.join(H8_CANDIDATES)}. Cặp H8 = ({h8['main']}, mức hợp lệ có r thấp nhất), tối đa {h8['max_pairs']} cặp. "
              "Không mức nào hợp lệ → H8 ra khỏi họ xác nhận (mô tả). Không train thêm detector.", "",
              "| mức | r (score chính) | CI 95 % | r_main − r | hợp lệ |", "|---|---|---|---|---|"]
        lv, key = rec.get("levels", {}), f"score>={sm}"
        for name in [h8["main"]] + H8_CANDIDATES:
            if name in lv:
                x, g = lv[name][key], h8["gaps"].get(name)
                ok = "chính" if name == h8["main"] else ("có" if name in h8["valid"] else "không")
                L.append(f"| {name} | {x['r']:.4f} | [{x['ci'][0]:.4f}; {x['ci'][1]:.4f}] | {'—' if g is None else f'{g:.4f}'} | {ok} |")
            else:
                L.append(f"| {name} | chưa có dump | — | — | chờ |")
        concl = {"pairs": f"cặp {', '.join('>'.join(p) for p in h8['pairs'])} vào họ xác nhận.",
                 "none_valid": "không mức nào hợp lệ → H8 ra khỏi họ xác nhận (chỉ mô tả).",
                 "pending": f"chưa mức nào hợp lệ; **H8 chờ {', '.join(h8['missing'])}** — dựng lại khi có r."}[h8["status"]]
        L += ["", "**Kết luận H8:** " + concl, ""]
    rd = df[(df.comparison == "r_pair") & df.eligible & df.r_dagger.notna()] if "r_dagger" in df else df.iloc[:0]
    if len(rd):
        rd = rd[rd.pool.isna()] if rd.pool.isna().any() else rd
        n_bad = int((rd.rdag_ok == False).sum())  # noqa: E712
        L += [f"- **r† cho H8 (P2H-Q7)**: r† = argmax_r Δ_pred(r) trên [0, 1] cho từng cấu hình (cue, θ*, tầng ego × M, L), cùng mô hình "
              "dự đoán của ô (Remark G2′: Δ lõm, Δ(0) = 0 ⇒ Δ không tăng trên [r†, 1]). Ô H8 chỉ giữ khi r_main > r† và r_low > r†; ngược lại "
              "dự đoán \"—\" và ra khỏi họ xác nhận. r† = 0 nghĩa là Δ_pred(r) không tăng trên toàn [0, 1].",
              f"  - r_main = {rd.r_hi.iloc[0]:.3f}, r_low = {rd.r_lo.iloc[0]:.3f}; r† lớn nhất = {rd.r_dagger.max():.3f}; "
              f"số ô H8 vi phạm: {n_bad}/{len(rd)} (ô H8 đủ cỡ mẫu, kể cả L = 10 đã bị thu gọn).", "",
              "| cue | ego | M_bin | L | r† | r_main > r† và r_low > r† |", "|---|---|---|---|---|---|"]
        for x in rd.sort_values(["cue", "ego", "M_bin", "L"]).itertuples(index=False):
            L.append(f"| {x.cue} | {x.ego} | {x.M_bin} | {int(x.L)} | {x.r_dagger:.3f} | {'có' if x.rdag_ok else '**không** → —'} |")
        L.append("")
    L += ["- **Luật chấm 3 mức (P2H-Q6, thay DRAFT §4 \"Scoring\")** — CI 95 % = bootstrap ghép cặp theo chuỗi, B = 1000, seed 42:",
          f"  - dự đoán \"+\": ĐÚNG iff CI_dưới(Δ̂) > 0; SAI iff CI_trên ≤ 0; còn lại CHƯA KẾT LUẬN.",
          f"  - dự đoán \"≤0\": ĐÚNG iff CI_trên < δ_min ({MARGIN:g}); SAI iff CI_dưới > δ_min; còn lại CHƯA KẾT LUẬN.",
          "  - H8: cùng luật áp lên ΔΔ̂. Iso-KPI (H2): ĐÚNG/SAI theo \"gate rẻ hơn ở KPI bằng nhau\"; CHƯA KẾT LUẬN khi CI của (a_G − a_P(ε)) chứa 0.",
          "  - Tiêu chí bác bỏ (i): < 80 % ô KẾT LUẬN ĐƯỢC là đúng; (ii) q_o* ngoài CI dự đoán ở ≥ 2/3 nhóm M; (iii) H8 sai ở ≥ 50 % ô H8 kết luận được.",
          "  - **Tiêu chí công suất P**: > 50 % ô họ xác nhận CHƯA KẾT LUẬN → kết luận bài = \"chưa kiểm được\" (không xác nhận, không bác bỏ); "
          "báo tỉ lệ 3 mức theo H2 / H3–H5 / H8. Holm chỉ để báo cáo. Code: `code/p2/p2_prereg_score.py`.", ""]
    # §3
    if hv:
        gm, gp, thr = hv["gmm2"], hv["largest_gap"], hv["proposed_threshold_px"]
        eg = P0 / "egomotion_seq.parquet"
        L += ["## §3 Ngưỡng hovering (ego-motion)", "",
              "- Biến: dịch tâm ảnh qua homography khung liền kề (ORB + RANSAC), **median theo chuỗi**, px/khung; mô hình trên **log10** của biến. "
              "Chỉ chuỗi TRAIN.",
              f"- Phương pháp chính: GMM 2 thành phần (EM, seed 42) trên log10 → ngưỡng = điểm hai thành phần (có trọng số) bằng nhau: "
              f"**{thr:.2f} px/khung**; mode {gm['mu_px'][0]:.3f} / {gm['mu_px'][1]:.3f} px, Ashman D = {gm['ashman_D']:.2f} (> 2 ⇒ bimodal) → "
              f"**{hv['n_hover_at_proposed']} hovering / {hv['n_seq'] - hv['n_hover_at_proposed']} moving** chuỗi.",
              f"- Đã thử: khe lớn nhất giữa hai giá trị liên tiếp (mỗi phía ≥ 3 chuỗi): khe {gp['gap_log10']:.2f} log10 (< 0,3) tại {gp['threshold_px']:.3f} px "
              f"({gp['n_below']}/{gp['n_above']} chuỗi) → **không bimodal theo khe, không dùng**.",
              "- Độ nhạy ngưỡng ±25 % = phụ lục mô tả (không vào họ xác nhận):"]
        if eg.exists():
            e = pd.read_parquet(eg)
            v = e[e.dataset == "UAVDT"].shift_px_median.values
            for fct in (0.75, 1.0, 1.25):
                L.append(f"  - ngưỡng × {fct:g} = {thr * fct:.2f} px → {int((v < thr * fct).sum())} hovering / {int((v >= thr * fct).sum())} moving.")
        L.append("")
    # §4
    if bc:
        ss = [x for x in channel.get("det_ms_sessions", []) if x.get("det_ms")]
        den = bc["t_ms"]["yolo26s_1024/GPU"]
        L += ["## §4 Chi phí cue c", "",
              f"- c = t_cue / t_det, **tử số và mẫu số cùng một phiên đo** (bench_c_real.json: {bc['n_pairs']} cặp khung TRAIN thật, {bc['n_seqs']} chuỗi, "
              f"{bc['repeats']} lần, median): mẫu số yolo26s@1024 iGPU (Ultralytics predict) = **{den['median']:.1f} ms** "
              f"(các lần: {', '.join(f'{x:.1f}' for x in den['runs'])} ms). Cue OpenCV đo **1 luồng**; tiny_det = yolo26n@320 CPU."]
        if ss:
            L.append(f"- Dải mẫu số yolo26s@1024 iGPU qua {len(ss)} phiên: **{min(x['det_ms'] for x in ss):.1f}–{max(x['det_ms'] for x in ss):.1f} ms** ("
                     + "; ".join(f"{x['session']}: {x['det_ms']:.1f}" for x in ss) + "). Chỉ báo dải; c dùng phiên cùng tử số.")
        L += ["- c theo cue (median): " + "; ".join(f"{k.split('/')[0]} {v['c_median']:.3f}" for k, v in bc["c"].items()
                                                     if k.endswith("/1thr") or k == "tiny_det_yolo26n_320/CPU") + ".", ""]
    # §5
    if au:
        tm, ru = au["table_main"], au["rule"]
        L += ["## §5 Audit khung trước onset — CHỈ MÔ TẢ, KHÔNG loại sự kiện (luật cố định, áp cho TEST sau freeze)", "",
              f"- Mỗi sự kiện E3: b = khung onset (khung VISIBLE đầu tiên); box GT của track ở khung b; dump {au['dump']} (score ≥ {ru['score_min']}) "
              f"ở {ru['k_pre']} khung b−{ru['k_pre']}…b−1; khớp = IoU ≥ {ru['iou_min']} với box GT khung b. b−{ru['k_pre']} < khung đầu → censored-start; "
              f"≤ {ru['k_late'] - 1}/{ru['k_pre']} khớp → true-birth. Với ≥ {ru['k_late']}/{ru['k_pre']} khớp: "
              f"**partial-entry** = border và (box khung b cách mép ảnh thật ≤ {ru['edge_px']} px hoặc track có GT ở b−{ru['k_pre']}…b−1 với out_of_view = 2); "
              "**visibility-transition** = track có GT ở các khung đó với occlusion = 2 (hoặc out_of_view = 2 khi interior); "
              "**annotation-late** (gán trễ THẬT) = không có dòng GT nào của track trong các khung đó; còn lại → true-birth.",
              "- **Không loại sự kiện nào** (P2G-Q5). Nếu annotation-late THẬT > 20 % interior trên TRAIN → ghi hạn chế ở §6, vẫn không loại.", "",
              "| TRAIN, E3 VISIBLE chính | partial-entry | visibility-transition | annotation-late | true-birth | censored-start | n | % annotation-late |",
              "|---|---|---|---|---|---|---|---|"]
        for t in ("border", "interior", "all"):
            r = tm[t]
            L.append(f"| {t} | {r['partial-entry']} | {r['visibility-transition']} | {r['annotation-late']} | {r['true-birth']} | "
                     f"{r['censored-start']} | {r['n']} | {pc(r['share_annotation_late'])} |")
        t5 = au.get("table_edge5")
        if t5:
            L += ["", f"Độ nhạy ngưỡng mép (P2H-Q9, chỉ mô tả): partial-entry khi cách mép ≤ 5 px thay vì ≤ {ru['edge_px']} px:", "",
                  "| TRAIN, mép ≤ 5 px | partial-entry | visibility-transition | annotation-late | true-birth | censored-start | n | % annotation-late |",
                  "|---|---|---|---|---|---|---|---|"]
            for t in ("border", "interior", "all"):
                r = t5[t]
                L.append(f"| {t} | {r['partial-entry']} | {r['visibility-transition']} | {r['annotation-late']} | {r['true-birth']} | "
                         f"{r['censored-start']} | {r['n']} | {pc(r['share_annotation_late'])} |")
        pl = au.get("placebo") or {}
        L += ["", f"- Placebo (box cùng cỡ đặt ngẫu nhiên ở khung {pl.get('frame')}, seed {pl.get('seed')}): khớp {pc(pl.get('hit_rate'))} "
              f"({pl.get('n')} sự kiện) ⇒ các khớp là vật thể thật, không do mật độ box."]
        vc = au.get("visual_check")
        if vc:
            L.append(f"- Kiểm phụ ({vc['kind']}): đúng nhãn 5 lớp {vc['n_agree']}/{vc['n_evaluable']} = {pc(vc['agreement'])}; "
                     f"nhị phân \"đã thấy xe trước onset\" {vc['n_agree_binary']}/{vc['n_evaluable']} = {pc(vc['agreement_binary'])}. Bất đồng: "
                     + ", ".join(f"{d['audit_id']} ({d['e3_type']}, tự động {d['label']} {d['n_match']:.0f}/{ru['k_pre']}, mắt {d['visual']})"
                                 for d in vc["disagreements"])
                     + " — không đổi nhãn tự động. Chi tiết: AUDIT_SHORT_E3.md.")
        L.append("")
    # §6
    ts = [s for s in info.get("theta_star", []) if not s.get("excluded") and s.get("theta") is not None]
    plus = conf[conf.sign_pred == "+"]
    L += ["## §6 Điều đã biết trên TRAIN trước freeze (quan sát TRAIN — TEST CHƯA MỞ)", ""]
    if ts:
        L += [f"- Kênh khởi phát yếu: J = q_in − q_out tại θ* ≤ {max(s['J'] for s in ts):.3f} ở mọi cue ("
              + ", ".join(f"{s['cue']} {s['J']:.3f}" for s in ts) + ").",
              f"- Ràng buộc a_G ≤ {A_MAX:g} chặn θ* (θ* ≠ argmax J không ràng buộc) ở: "
              + (", ".join(f"{s['cue']} (a_G = {s['a_G']:.3f})" for s in ts if s.get("binding")) or "không cue nào") + "."]
    if len(plus):
        L.append(f"- Dự đoán gate thắng (\"+\") chỉ ở {len(plus)}/{len(conf)} ô họ xác nhận: "
                 + "; ".join(sorted({f"{x.hypothesis} {x.cue} × {x.ego} × M {x.M_bin} × {x.kpi}" for x in plus.itertuples()})) + ".")
    else:
        L.append(f"- Không ô nào trong {len(conf)} ô họ xác nhận dự đoán gate thắng (\"+\").")
    if au and au.get("limitation_flag_gt_20pct_interior"):
        L.append(f"- **Hạn chế dữ liệu (không loại sự kiện)**: {pc(au['interior_share_annotation_late'])} sự kiện E3 interior (TRAIN) là annotation-late THẬT "
                 "(> 20 %): detector đã thấy xe ≥ 3/5 khung trước onset mà track chưa có dòng GT nào; border: "
                 f"{pc(au['table_main']['border']['share_annotation_late'])}. Onset GT của các sự kiện này có thể muộn hơn lúc xe thật sự xuất hiện "
                 "→ xem phụ lục onset_alt.")
    L += ["- **Bản 01-10-2026 sáng (P2F: lọc annotation-late, loại 82 % E3 TRAIN) bị RÚT vì luật audit sai** (gộp xe vào từ mép lộ một phần "
          "và track đã có GT với occlusion/out_of_view = 2 — chính định nghĩa onset của P0 — thành \"gán trễ\"); **không số nào từ bản đó được dùng**.",
          "- Mọi điều trên là quan sát trên TRAIN dùng để dựng dự đoán; TEST chưa mở.", ""]
    if "delta_pred_onset_alt" in conf:
        mc = conf[(conf.comparison == "matched_cost") & conf.delta_pred_onset_alt.notna()]
        if len(mc):
            sh = au.get("onset_alt", {}) if au else {}
            L += ["## Phụ lục — độ nhạy onset (mô tả, không vào họ xác nhận)", "",
                  f"onset_alt = khung sớm nhất trong b−5…b−1 có detector khớp (không có → b). TRAIN: {sh.get('n_shifted')} sự kiện dời sớm, "
                  f"trung bình {sh.get('shift_mean', float('nan')):.2f} khung. Dự đoán matched cost (H3/H5) ở ô xác nhận, onset GT vs onset_alt "
                  "(cửa sổ KPI T = min(D + shift, L + 1)):", "",
                  "| cue | ego | M_bin | L | Δ_pred onset GT | Δ_pred onset_alt | thay đổi | dự đoán GT → alt |", "|---|---|---|---|---|---|---|---|"]
            for x in mc.sort_values(["cue", "ego", "M_bin", "L"]).itertuples(index=False):
                s_alt = "+" if x.delta_pred_onset_alt > MARGIN else "<=0"
                L.append(f"| {x.cue} | {x.ego} | {x.M_bin} | {int(x.L)} | {x.delta_pred:+.4f} | {x.delta_pred_onset_alt:+.4f} | "
                         f"{x.delta_pred_onset_alt - x.delta_pred:+.4f} | {x.sign_pred} → {s_alt} |")
            L.append("")
    return L


def ego_labels(dataset="uavdt"):
    """{seq: 'hovering'|'moving'} theo ngưỡng ĐỀ XUẤT TỪ PHÂN BỐ (results/p0/hover_threshold.json, cùng luật p2_estimate_channel);
    {} nếu chưa có / unimodal. P2F: bản cũ đọc cột `ego` của egomotion_seq.parquet (ngưỡng tạm 1,0 px → 24/6 chuỗi) — sai."""
    eg, ht = P0 / "egomotion_seq.parquet", P0 / "hover_threshold.json"
    if not eg.exists() or not ht.exists():
        return {}
    thr = json.loads(ht.read_text(encoding="utf-8")).get("by_dataset", {}).get(dataset.upper(), {}).get("proposed_threshold_px")
    if thr is None:
        return {}
    e = pd.read_parquet(eg)
    e = e[e.dataset.str.lower() == dataset] if "dataset" in e else e
    return dict(zip(e.seq, np.where(e.shift_px_median < thr, "hovering", "moving")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=str(ROOT / "results" / "p2" / "channel_train.json"))
    ap.add_argument("--out-md", default=str(ROOT / "PREREG_28.md"))
    ap.add_argument("--out-json", default=str(ROOT / "results" / "p2" / "prereg_cells.json"))
    ap.add_argument("--mode", default="meanfield", choices=["meanfield", "asparse"])
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    channel = json.loads(Path(a.channel).read_text(encoding="utf-8"))
    info = {}
    cells = build(channel, train_events(), ego_by_seq=ego_labels(), info=info)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(dict(channel_source=a.channel, simulated=channel.get("simulated"), n=len(cells), info=info, cells=cells),
                                           indent=1, default=float), encoding="utf-8")
    write_md(cells, channel, a.out_md, info)
    print("cells", len(cells), "confirmatory", info["family"]["n_final"], "→", a.out_json, a.out_md)


if __name__ == "__main__":
    main()
