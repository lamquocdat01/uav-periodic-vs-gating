"""P1: kiểm số cho THEORY_28.md bằng công thức đóng + Monte Carlo (seed=42).

Mỗi kiểm tra ghi: tên, phát biểu (THEORY_28 §), số mẫu, sai lệch lớn nhất theo σ (z), PASS/FAIL (|z| ≤ 3
cho MC; sai số ≤ 1e-12 cho đồng nhất thức). Đầu ra: results/p1/theory_checks.json.
Số minh hoạ (ρ_onset, M trung vị UAVDT) đọc từ results/p0_stats.json nếu có.
Chạy: python code/theory_checks.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p1"
OUT.mkdir(parents=True, exist_ok=True)
STATS = ROOT / "results" / "p0_stats.json"
SEED = 42
rng = np.random.default_rng(SEED)
CHECKS = []


def record(name, section, n, max_z=None, max_err=None, passed=None, zlist=None, **extra):
    if zlist is not None:
        # họ nhiều phép so: ngoài max |z| (tiêu chí 3σ từng phép) ghi thêm kiểm định gộp χ²(k) và số vượt 3σ kỳ vọng
        from scipy.stats import chi2
        z = np.asarray([x for x in zlist if np.isfinite(x)])
        extra.update(n_comparisons=len(zlist), n_exceed_3sigma=int(np.sum(np.asarray(zlist) > 3)),
                     expected_exceed_3sigma_if_true=float(len(zlist) * 0.0026998),
                     chi2_sum_z2=float(np.sum(z ** 2)), chi2_p=float(chi2.sf(np.sum(z ** 2), len(z))) if len(z) else None,
                     n_nonfinite=int(len(zlist) - len(z)))
        max_z = float(max(zlist))
    if passed is None:
        passed = (max_z is None or max_z <= 3.0) and (max_err is None or max_err <= 1e-12)
    CHECKS.append(dict(name=name, section=section, n=n, max_z=max_z, max_err=max_err, passed=bool(passed), **extra))
    print(f"[{'PASS' if passed else 'FAIL'}] {section} {name} n={n} max_z={max_z} max_err={max_err}",
          {k: v for k, v in extra.items() if k.startswith(("n_comp", "n_exceed", "expected", "chi2_p"))})


def z_bin(ind, p):
    """z cho biến chỉ báo với SE nhị thức theo xác suất lý thuyết p (tránh σ mẫu = 0 khi p ≈ 0 hoặc 1)."""
    n = len(ind)
    m = float(np.mean(ind))
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    if se < 1e-12:
        return 0.0 if abs(m - p) < 1e-9 else float("inf")
    # p rất gần 0/1: sai lệch dưới độ phân giải 1/n không tính là lệch
    return max(0.0, abs(m - p) - 0.5 / n) / se


def z_of(sample, want):
    m = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / math.sqrt(len(sample)))
    if se < 1e-12:  # mẫu hằng (trường hợp tất định): so trực tiếp, dung sai làm tròn
        return 0.0 if abs(m - want) < 1e-9 else float("inf")
    return abs(m - want) / se


# ======================================================================= §2 Lemma 1
def lemma_miss(D, S, r):
    """miss_r(D,S) = (1−r)^n (1 − f r), n = ⌊D/S⌋, f = frac(D/S). D, S thực > 0."""
    x = np.asarray(D, float) / S
    n = np.floor(x + 1e-12)
    f = np.clip(x - n, 0.0, 1.0)
    q = 1.0 - r
    return np.where(n == 0, 1.0 - f * r, (q ** n) * (1.0 - f * r))


def looks_discrete(D, S, phase):
    """Lịch tuần hoàn stride thực S ≥ 1 trên khung rời rạc: nhìn khung ⌈φ + kS⌉, φ ~ U[0,S).
    Sự kiện chiếm khung {0,…,D−1}. Trả số lần nhìn rơi vào sự kiện."""
    # k chạy từ k_min sao cho ⌈φ+kS⌉ ≥ 0 tới khi > D−1
    kmax = int(np.ceil((D + 1) / S)) + 2
    k = np.arange(-2, kmax)
    t = np.ceil(phase[:, None] + k[None, :] * S - 1e-12)
    return ((t >= 0) & (t <= D - 1)).sum(axis=1)


def check_lemma1():
    n_mc = 100_000
    cases = [(1, 5), (3, 10), (7, 5), (12, 10), (25, 12), (8, 8), (13, 3), (2, 240), (5, 2.5), (9, 3.7), (4, 1.6), (30, 7.25)]
    zs = []
    for D, S in cases:
        ph = rng.uniform(0, S, n_mc)
        N = looks_discrete(D, S, ph)
        n0, f = math.floor(D / S), D / S - math.floor(D / S)
        # luật N: P(N = n0+1) = f, P(N = n0) = 1 − f, không giá trị khác
        support_ok = bool(np.all((N == n0) | (N == n0 + 1)))
        zs.append(z_of((N == n0 + 1).astype(float), f))
        for r in (0.5, 0.8, 1.0):
            zs.append(z_of((1 - r) ** N, float(lemma_miss(D, S, r))))
        if not support_ok:
            zs.append(float("inf"))
    record("N = ⌊D/S⌋ + Bernoulli(frac(D/S)); miss_r = (1−r)^n(1−fr) (S nguyên và thực)", "§2 Lemma 1", n_mc * len(cases), zlist=zs)
    # r=1 ⇒ (1 − D/S)_+ chính xác
    D = np.arange(1, 301)
    err = max(float(np.max(np.abs(lemma_miss(D, S, 1.0) - np.clip(1 - D / S, 0, None)))) for S in (1, 2, 3, 5, 7.5, 10, 50, 120, 240))
    record("r=1 ⇒ miss = (1 − D/S)_+", "§2 Lemma 1(ii)", 300 * 9, max_err=err)
    # đơn điệu không tăng theo D (liên tục ở D = nS)
    worst = 0.0
    for S in (1, 2, 3, 5, 10, 25.5, 120):
        for r in (0.3, 0.5, 0.8, 1.0):
            v = lemma_miss(np.linspace(0.01, 400, 40001), S, r)
            worst = max(worst, float(np.max(np.diff(v))))
    record("miss_r(D,S) không tăng theo D", "§2 Lemma 2", 7 * 4 * 40001, max_err=max(worst, 0.0), passed=worst <= 1e-12)


def check_censoring():
    """Cửa sổ sinh: r=1 chính xác; r<1 cận trên; KM gần đúng. Mô phỏng dữ liệu kiểu UAVDT."""
    W, L = 120, 1200
    reps = 400
    diff_r1, gap_r05, km_z = [], [], []
    truth_r05, km_r05 = [], []
    S = 50
    for _ in range(reps):
        nb = rng.poisson(0.04 * L)
        b = rng.integers(0, L, nb)
        # hỗn hợp: 8% đuôi ngắn (1–7), còn lại geometric mean 250
        short = rng.random(nb) < 0.08
        D = np.where(short, rng.integers(1, 8, nb), 1 + rng.geometric(1 / 250, nb))
        end = np.minimum(b + D - 1, L - 1)
        Dobs = end - b + 1
        cens = b + D - 1 > L - 1
        win = b <= L - 1 - W
        if win.sum() == 0:
            continue
        # r=1: ước lượng cửa sổ trên D_obs = trên D thật (từng sự kiện)
        diff_r1.append(float(np.max(np.abs(lemma_miss(Dobs[win], S, 1.0) - lemma_miss(D[win], S, 1.0)))))
        gap_r05.append(float(np.mean(lemma_miss(Dobs[win], S, 0.5)) - np.mean(lemma_miss(D[win], S, 0.5))))
    record("r=1: ước lượng cửa sổ sinh (D_obs) = D thật cho mọi S ≤ W", "§2 Cor. 1", reps, max_err=max(diff_r1))
    record("r<1: ước lượng cửa sổ sinh là cận trên (sai lệch ≥ 0)", "§2 Cor. 1(ii)", reps, max_err=0.0, passed=min(gap_r05) >= -1e-12,
           min_gap=min(gap_r05), mean_gap=float(np.mean(gap_r05)))


# ======================================================================= §3 P1 kinematic + latency
def check_P1():
    """Xe điểm đi thẳng qua vệt nhìn độ dài L với v ≤ v_max; detector chu kỳ τ = S/F trên khung camera F fps.
    Phát biểu: f_det = F/S ≥ n·v_max/L ⇒ N ≥ n cho MỌI pha ⇒ miss ≤ (1−r)^n ≤ ε. Không phụ thuộc F."""
    theta, h, vmax, eps = math.radians(60), 100.0, 25.0, 0.01
    L = 2 * h * math.tan(theta / 2)
    res = {}
    min_ok = True
    zs = []
    for r in (0.5, 0.8, 0.95):
        n = math.ceil(math.log(eps) / math.log(1 - r))
        fdet_req = n * vmax / L
        for F in (10.0, 15.0, 30.0, 60.0, 120.0):
            S = math.floor(F / fdet_req)  # stride nguyên lớn nhất bảo đảm F/S ≥ f_det_req
            if S < 1:
                continue
            m = 20_000
            v = rng.uniform(0.2 * vmax, vmax, m)
            t0 = rng.uniform(0, 1000, m)  # thời điểm vào vệt (giây), pha tuỳ ý
            phase = rng.integers(0, S, m)
            # khung camera t_j = j/F; detector tại j ≡ phase (mod S); xe thấy trong [t0, t0 + L/v)
            j0 = np.ceil(t0 * F - 1e-9).astype(np.int64)
            j1 = np.ceil((t0 + L / v) * F - 1e-9).astype(np.int64) - 1
            first = j0 + ((phase - j0) % S)
            N = np.where(first <= j1, (j1 - first) // S + 1, 0)
            min_ok &= bool(N.min() >= n)
            hit = rng.random((m, int(N.max()))) < r
            valid = np.arange(int(N.max()))[None, :] < N[:, None]
            missed = ~(hit & valid).any(axis=1)
            res[f"r{r}|F{F:g}"] = dict(n=n, S=S, f_det=F / S, f_det_req=fdet_req, N_min=int(N.min()), miss_mc=float(missed.mean()))
            # miss ≤ ε: kiểm một phía — MC không vượt ε quá 3σ
            se = math.sqrt(max(eps * (1 - eps), 1e-12) / m)
            zs.append(max(0.0, (missed.mean() - eps) / se))
    record("f_det ≥ n·v_max/L ⇒ N_min ≥ n (mọi pha) và miss ≤ ε, với F ∈ {10,…,120} fps", "§3 Thm P1", 15 * 20_000,
           max_z=max(zs), passed=min_ok and max(zs) <= 3, detail=res, L_m=L)
    # biến thể latency: P(L ≤ Lmax) = 1 − miss_r(min(D, Lmax+1), S); = 1 nếu r=1, S ≤ Lmax+1, D ≥ S
    zs, exact_one = [], True
    n_mc = 50_000
    for D in (2, 4, 7, 12, 40):
        for S in (1, 2, 3, 4, 6, 11, 20):
            for Lm in (3, 5, 10):
                for r in (1.0, 0.8, 0.5):
                    ph = rng.integers(0, S, n_mc)
                    # lần nhìn tại offset ph + kS (k ≥ 0) trong sự kiện; L = offset lần thành công đầu
                    kmax = D // S + 2
                    offs = ph[:, None] + S * np.arange(kmax)[None, :]
                    inev = offs <= D - 1
                    succ = inev & (rng.random(offs.shape) < r)
                    Lfirst = np.where(succ.any(axis=1), np.where(succ, offs, 10 ** 9).min(axis=1), 10 ** 9)
                    ok = (Lfirst <= Lm).astype(float)
                    want = 1.0 - float(lemma_miss(min(D, Lm + 1), S, r))
                    zs.append(z_bin(ok, want))
                    if r == 1.0 and S <= Lm + 1 and D >= S:
                        exact_one &= bool(ok.min() == 1.0) and abs(want - 1.0) < 1e-12
    record("P(L ≤ Lmax) = 1 − miss_r(min(D,Lmax+1),S); =1 khi r=1, S ≤ Lmax+1, D ≥ S", "§3 Cor. P1-lat", 5 * 7 * 3 * 3 * n_mc,
           zlist=zs, passed=max(zs) <= 3 and exact_one, exact_one_r1=exact_one)


# ======================================================================= §4 P2 độ cao tối ưu
def check_P2():
    out = {}
    # ví dụ log-logistic theo cỡ pixel s(h) = kappa/h: r = r_max / (1 + (s0/s)^β)
    kappa, s0, rmax = 1500.0, 12.0, 0.9  # 30 px ở 50 m
    h = np.linspace(1, 2000, 400_000)
    for beta in (0.8, 1.0, 1.5, 3.0):
        r = rmax / (1 + (s0 * h / kappa) ** beta)
        g = -np.log(1 - r)
        psi = h * g
        i = int(np.argmax(psi))
        interior = 0 < i < len(h) - 1 and psi[-1] < psi[i] * 0.999
        rec = dict(beta=beta, argmax_h=float(h[i]), interior=bool(interior), psi_end_over_max=float(psi[-1] / psi[i]))
        if beta > 1:
            # FOC chính xác: d ln g / d ln h = −1 tại h*
            dlng = np.gradient(np.log(g), np.log(h))
            j = int(np.argmin(np.abs(dlng[1000:-1000] + 1))) + 1000
            rec.update(foc_h=float(h[j]), approx_h_small_r=float(kappa / s0 * (beta - 1) ** (-1 / beta)))
        out[f"loglogistic_beta{beta}"] = rec
    ok_ll = (out["loglogistic_beta1.5"]["interior"] and out["loglogistic_beta3.0"]["interior"]
             and not out["loglogistic_beta0.8"]["interior"]
             and abs(out["loglogistic_beta3.0"]["foc_h"] - out["loglogistic_beta3.0"]["argmax_h"]) / out["loglogistic_beta3.0"]["argmax_h"] < 0.01)
    # logistic theo s có sàn r0 > 0 khi s → 0: không có cực tiểu toàn cục hữu hạn (phản ví dụ)
    w = 3.0
    r = rmax / (1 + np.exp(-((kappa / h) - s0) / w))
    psi = h * -np.log(1 - r)
    out["logistic_floor"] = dict(r_floor=float(rmax / (1 + math.exp(s0 / w))), psi_increasing_tail=bool(np.all(np.diff(psi[-1000:]) > 0)),
                                 local_max_h=float(h[int(np.argmax(psi[: len(h) // 4]))]),
                                 psi_local_max=float(psi[: len(h) // 4].max()), psi_at_2000=float(psi[-1]))
    ok_floor = out["logistic_floor"]["psi_increasing_tail"]
    # bản nguyên: f(h) ∝ ⌈ln ε / ln(1−r(h))⌉ / h có cực tiểu (hàm nửa liên tục dưới, → ∞ ở hai đầu)
    eps = 0.01
    beta = 3.0
    r = rmax / (1 + (s0 * h / kappa) ** beta)
    nint = np.ceil(np.log(eps) / np.log(1 - r) - 1e-12)
    f = nint / h
    k = int(np.argmin(f))
    out["integer_n"] = dict(argmin_h=float(h[k]), n_at_min=int(nint[k]), f_min=float(f[k]), f_ends=[float(f[0]), float(f[-1])])
    ok_int = 0 < k < len(h) - 1
    record("P2: β>1 ⇒ h* nội tại, FOC d ln g/d ln h = −1; β ≤ 1 hoặc sàn r0>0 ⇒ không có h* hữu hạn", "§4 Thm P2 + phản ví dụ",
           len(h), max_err=0.0, passed=ok_ll and ok_floor and ok_int, detail=out)
    return out


# ======================================================================= §5 kênh gate
def a_gate(rho, q_in, q_out, R):
    fR = 0.0 if R == math.inf else 1.0 / R
    return fR + (1 - fR) * (rho * q_in + (1 - rho) * q_out)


_PH = {}


def _phase_counts(T, w, R):
    """Với refresh pha đều chu kỳ R: số refresh trong w' khung đầu (n_in) và phần còn lại (n_out), mỗi pha."""
    key = (T, w, R)
    if key not in _PH:
        wp = min(w, T)
        ph = np.arange(int(R))[:, None]
        t = np.arange(T)[None, :]
        ref = (t - ph) % int(R) == 0
        _PH[key] = (ref[:, :wp].sum(axis=1), ref[:, wp:].sum(axis=1))
    return _PH[key]


def miss_gate(T, w, R, q_in, q_out, r):
    """Miss của gate+refresh trên cửa sổ KPI T khung; w' = min(w,T) khung đầu là cửa sổ khởi phát (q_in),
    phần còn lại q_out (giả định A-sparse); refresh pha đều chu kỳ R (R=inf: không refresh).
    q_in có thể là mảng (vector hoá)."""
    wp = min(w, T)
    q_in = np.asarray(q_in, float)
    if R == math.inf:
        return (1 - r * q_in) ** wp * (1 - r * q_out) ** (T - wp)
    n_in, n_out = _phase_counts(T, w, R)
    v = ((1 - r) ** (n_in + n_out))[None, :] * (1 - r * q_in[..., None]) ** (wp - n_in)[None, :]         * ((1 - r * q_out) ** (T - wp - n_out))[None, :]
    out = v.mean(axis=-1)
    return out if out.shape != (1,) or np.ndim(q_in) else float(out[0])


def delta(T, w, R, q_in, q_out, rho, r, c=0.0):
    """Δ = miss_periodic(cùng chi phí a_G + c) − miss_gate. q_in có thể là mảng."""
    q_in = np.asarray(q_in, float)
    a = np.minimum(a_gate(rho, q_in, q_out, R) + c, 1.0)
    mp = lemma_miss(T, 1.0 / a, r)
    res = mp - miss_gate(T, w, R, q_in, q_out, r)
    return float(res) if np.ndim(res) == 0 or (np.ndim(q_in) == 0) else res


def check_gate():
    # (1) activation: mô phỏng chuỗi khung với sự kiện Poisson, cửa sổ w, gate Bernoulli, refresh pha đều
    zs = []
    for (lam, w, R, qi, qo) in [(0.04, 5, 30, 0.6, 0.02), (0.02, 3, 10, 0.9, 0.1), (0.08, 10, math.inf, 0.3, 0.05), (0.04, 5, 5, 0.5, 0.2)]:
        Lf, reps = 20_000, 30
        acts, rho_hat = [], []
        for _ in range(reps):
            I = np.zeros(Lf, bool)
            for b in np.flatnonzero(rng.random(Lf) < lam):
                I[b:b + w] = True
            G = rng.random(Lf) < np.where(I, qi, qo)
            ref = np.zeros(Lf, bool)
            if R != math.inf:
                ref[int(rng.integers(0, R))::int(R)] = True
            run = G | ref
            acts.append(run.mean())
            rho_hat.append(I.mean())
        # so với công thức dùng ρ thực nghiệm từng lần lặp
        want = np.array([a_gate(rh, qi, qo, R) for rh in rho_hat])
        zs.append(z_of(np.array(acts) - want, 0.0))
    record("a_G = 1/R + (1−1/R)[ρ q_in + (1−ρ) q_out]", "§5 Prop. G1", 4 * 30 * 20_000, zlist=zs)
    # (2) miss gate (một sự kiện) MC vs công thức, r ∈ {1, 0.8, 0.5}
    zs = []
    n_mc = 60_000
    for (T, w, R, qi, qo) in [(4, 3, 10, 0.5, 0.05), (6, 5, 30, 0.8, 0.02), (11, 5, math.inf, 0.3, 0.1), (6, 6, 4, 0.2, 0.3), (31, 5, 20, 0.7, 0.01)]:
        for r in (1.0, 0.8, 0.5):
            wp = min(w, T)
            ph = rng.integers(0, int(R), n_mc) if R != math.inf else None
            t = np.arange(T)[None, :]
            ref = ((t - ph[:, None]) % int(R) == 0) if R != math.inf else np.zeros((n_mc, T), bool)
            q = np.where(t < wp, qi, qo)
            run = ref | (rng.random((n_mc, T)) < q)
            succ = run & (rng.random((n_mc, T)) < r)
            missed = (~succ.any(axis=1)).astype(float)
            zs.append(z_of(missed, miss_gate(T, w, R, qi, qo, r)))
    record("miss_G (gate ∨ refresh, A-ind, A-sparse) khớp MC", "§5 Prop. G2", 15 * n_mc, zlist=zs)
    # (3) đồng nhất thức lợi–phí (r=1): Δ = G − C − T c khi T(a_G + c) ≤ 1 và T ≤ R
    err = 0.0
    n = 0
    for _ in range(20_000):
        T = int(rng.integers(1, 40))
        w = int(rng.integers(1, T + 1))
        R = int(rng.integers(T, 200))
        qi, qo, rho, c = rng.random(), rng.random() * 0.2, rng.random() * 0.5, rng.random() * 0.02
        a = a_gate(rho, qi, qo, R)
        if T * (a + c) > 1:
            continue
        G = (1 - T / R) * (1 - (1 - qi) ** w * (1 - qo) ** (T - w))
        C = T * (1 - 1 / R) * (rho * qi + (1 - rho) * qo)
        err = max(err, abs(delta(T, w, R, qi, qo, rho, 1.0, c) - (G - C - T * c)))
        n += 1
    record("Δ = G − C − T·c (r=1)", "§5 Thm G3", n, max_err=err, passed=err <= 1e-10)
    # (4) tập thắng theo q_in là một khoảng; điều kiện cần q_out < (κ − cR/(R−1))/(1−ρ) (r=1)
    viol, nonint, n = 0, 0, 0
    for _ in range(3000):
        T = int(rng.integers(2, 32))
        w = int(rng.integers(1, T + 1))
        R = int(rng.integers(T + 1, 300)) if rng.random() < 0.8 else math.inf
        qo, rho, c = rng.random() ** 2, rng.random() * 0.6, rng.random() * 0.03 * (rng.random() < 0.5)
        qs = np.linspace(0, 1, 201)
        d = np.atleast_1d(delta(T, w, R, qs, qo, rho, 1.0, c))
        win = d > 1e-12
        if win.any():
            idx = np.flatnonzero(win)
            if not np.all(win[idx[0]:idx[-1] + 1]):
                nonint += 1
            kap = (1 - T / R) / (T * (1 - 1 / R)) if R != math.inf else 1 / T
            cR = c * (R / (R - 1) if R != math.inf else 1.0)
            if not ((1 - rho) * qo + cR < kap):
                viol += 1
        n += 1
    record("r=1: tập q_in thắng là một khoảng; mọi cấu hình thắng thoả điều kiện cần (N-G)", "§5 Cor. G4", n, max_err=0.0,
           passed=viol == 0 and nonint == 0, n_nonconvex=nonint, n_violate_necessary=viol)
    # (5) mô phỏng trọn chuỗi (bỏ A-sparse): dấu Δ dự đoán khớp dấu MC ở cấu hình thắng rõ và thua rõ
    res = {}
    zs = []
    for name, (lam, w, T, R, qi, qo) in {"win": (0.01, 3, 4, 60, 0.9, 0.005), "lose": (0.04, 5, 6, 30, 0.6, 0.15)}.items():
        Lf, reps = 30_000, 20
        dm = []
        for _ in range(reps):
            births = np.flatnonzero(rng.random(Lf - T) < lam)
            I = np.zeros(Lf, bool)
            for b in births:
                I[b:b + w] = True
            G = rng.random(Lf) < np.where(I, qi, qo)
            ref = np.zeros(Lf, bool)
            ref[int(rng.integers(0, R))::R] = True
            run = G | ref
            a = run.mean()
            S = 1.0 / a
            ph = rng.uniform(0, S)
            per = np.zeros(Lf, bool)
            per[np.unique(np.ceil(ph + S * np.arange(int(Lf / S) + 2)).astype(int).clip(0, Lf - 1))] = True
            mg = np.mean([not run[b:b + T].any() for b in births])
            mpp = np.mean([not per[b:b + T].any() for b in births])
            dm.append(mpp - mg)
        rho_hat = lam * w  # xấp xỉ thưa để tính dự đoán
        pred = delta(T, w, R, qi, qo, rho_hat, 1.0)
        res[name] = dict(delta_mc=float(np.mean(dm)), delta_se=float(np.std(dm, ddof=1) / math.sqrt(reps)), delta_pred_sparse=float(pred))
    sign_ok = res["win"]["delta_mc"] > 0 and res["lose"]["delta_mc"] < 0 and res["win"]["delta_pred_sparse"] > 0 and res["lose"]["delta_pred_sparse"] < 0
    record("Chuỗi đầy đủ (không A-sparse): dấu Δ khớp dự đoán", "§5 kiểm toàn chuỗi", 2 * 20 * 30_000, max_err=0.0, passed=sign_ok, detail=res)


# ======================================================================= §5 Cor. G5 iso-KPI (r=1, gate khởi phát lý tưởng w=1, R=∞)
def check_isoKPI():
    """Đạt latency-miss ≤ ε cho sự kiện D ≥ T: periodic cần a_P = (1−ε)/T; gate khởi phát (q_in=1 ở khung đầu)
    đạt miss 0 với chi phí a_G + c = ρ1 + (1−ρ1) q_out + c. Mô phỏng chuỗi đầy đủ."""
    res, zs = {}, []
    for (lam, T, qo, eps) in [(0.04, 4, 0.05, 0.05), (0.02, 11, 0.02, 0.01), (0.04, 6, 0.10, 0.02)]:
        Lf, reps = 40_000, 20
        a_g, m_g, a_p_list, m_p = [], [], [], []
        for _ in range(reps):
            births = np.flatnonzero(rng.random(Lf - T) < lam)
            I1 = np.zeros(Lf, bool)
            I1[births] = True
            run = I1 | (rng.random(Lf) < qo)
            a_g.append(run.mean())
            m_g.append(np.mean([not run[b:b + T].any() for b in births]))
            rho1 = I1.mean()
            # periodic với a_P = (1−ε)/T, pha đều, stride thực
            S = T / (1 - eps)
            ph = rng.uniform(0, S)
            per = np.zeros(Lf, bool)
            per[np.ceil(ph + S * np.arange(int(Lf / S) + 2)).astype(int).clip(0, Lf - 1)] = True
            m_p.append(np.mean([not per[b:b + T].any() for b in births]))
            a_p_list.append(rho1 + (1 - rho1) * qo)
        zs.append(z_of(np.array(a_g) - np.array(a_p_list), 0.0))
        zs.append(z_of(np.array(m_p), eps))
        res[f"lam{lam}|T{T}|qo{qo}|eps{eps}"] = dict(a_gate_mc=float(np.mean(a_g)), a_gate_formula=float(np.mean(a_p_list)),
                                                      miss_gate_mc=float(np.mean(m_g)), a_periodic=(1 - eps) / T, miss_periodic_mc=float(np.mean(m_p)))
    ok0 = all(v["miss_gate_mc"] == 0.0 for v in res.values())
    record("iso-KPI: periodic cần a=(1−ε)/T; gate khởi phát lý tưởng: a=ρ1+(1−ρ1)q_out, miss=0", "§5 Cor. G5", 3 * 20 * 40_000,
           zlist=zs, passed=max(zs) <= 3 and ok0, detail=res)


# ======================================================================= §5 Remark G2' (multi-look, r < 1)
def r_dagger(T, w, q_in, q_out, a):
    """Đỉnh của Δ(r) = (1 − T a r) − (1 − r q_in)^{w'} (1 − r q_out)^{T−w'} trên [0,1] (vùng T a < 1). Tìm số (lưới mịn)."""
    wp = min(w, T)
    r = np.linspace(0, 1, 100_001)
    d = (1 - T * a * r) - (1 - r * q_in) ** wp * (1 - r * q_out) ** (T - wp)
    return float(r[int(np.argmax(d))]), d


def check_multilook():
    """(1) MC: Δ(r) theo công thức khớp mô phỏng một sự kiện (periodic lattice pha đều + gate Bernoulli);
    (2) Δ(r) lõm, Δ(0)=0 (lưới); (3) q_out = 0, w' = T: r† = (1 − (a/q_in)^{1/(T−1)})/q_in đóng; (4) phản ví dụ vô điều kiện: r < r† ⇒ Δ giảm khi r giảm."""
    zs, conc_ok, closed_err, n_cfg, below, cond_ok = [], True, 0.0, 0, 0, True
    n_mc = 40_000
    for (T, w, q_in, q_out, a) in [(4, 3, 1.0, 0.0, 0.09), (6, 5, 0.8, 0.02, 0.1), (4, 4, 0.5, 0.0, 0.05), (11, 10, 1.0, 0.01, 0.06), (6, 6, 1.0, 0.0, 0.12)]:
        assert T * a < 1
        S = 1 / a
        for r in (1.0, 0.8, 0.5, 0.3):
            ph = rng.uniform(0, S, n_mc)
            nP = looks_discrete(T, S, ph)
            mP = (rng.random((n_mc, int(nP.max()) + 1)) < r)
            detP = (mP[:, :] & (np.arange(mP.shape[1])[None, :] < nP[:, None])).any(axis=1)
            wp = min(w, T)
            q = np.r_[np.full(wp, q_in), np.full(T - wp, q_out)]
            detG = ((rng.random((n_mc, T)) < q[None, :]) & (rng.random((n_mc, T)) < r)).any(axis=1)
            want = (1 - T * a * r) - (1 - r * q_in) ** wp * (1 - r * q_out) ** (T - wp)
            zs.append(z_of(detG.astype(float) - detP.astype(float), want))
        rd, d = r_dagger(T, w, q_in, q_out, a)
        conc_ok &= bool(np.all(np.diff(d, 2) <= 1e-12) and abs(d[0]) < 1e-15)
        if q_out == 0 and min(w, T) == T:
            closed = min(1.0, (1 - (a / q_in) ** (1 / (T - 1))) / q_in)
            closed_err = max(closed_err, abs(closed - rd))
        # điều kiện r† < 1 ⇔ Δ'(1) < 0 ⇔ T a > −miss_G'(1)
        wp = min(w, T)
        dmG1 = -((1 - q_in) ** wp * (1 - q_out) ** (T - wp)) * (wp * q_in / max(1 - q_in, 1e-300) + (T - wp) * q_out / (1 - q_out))             if q_in < 1 else (-(T - wp) * 0.0 if wp >= 2 else -(1 - q_out) ** (T - 1))
        cond = T * a > -dmG1
        cond_ok &= (cond == (rd < 1 - 1e-4))
        below += int(0.05 < rd < 1 - 1e-4)   # đỉnh nội tại: dưới r† Δ giảm khi r giảm → không vô điều kiện
        n_cfg += 1
    record("Remark G2': Δ(r) khớp MC; Δ lõm, Δ(0)=0; r† đóng (cắt ở 1) khi q_out=0,w'=T; điều kiện r†<1 ⇔ Ta > −miss_G'(1)", "§5 Remark G2'",
           n_mc * len(zs), zlist=zs, passed=max(zs) <= 3 and conc_ok and closed_err < 1e-4 and cond_ok,
           concave_ok=conc_ok, r_dagger_closed_max_err=closed_err, r_dagger_condition_ok=cond_ok,
           n_cfg=n_cfg, n_cfg_with_interior_peak=below,
           note="cấu hình q_in=0.5 (T=4, a=0.05) có r†=1: Δ giảm khi r giảm trên cả [0,1] → Remark chỉ đúng có điều kiện")


def M_iso(T, eps, lam, q_b, q_o, c=0.0):
    """M lớn nhất (liên tục) để gate khởi phát lý tưởng còn rẻ hơn periodic ở cùng latency-miss ε (r=1)."""
    Q = ((1 - eps) / T - lam - c) / (1 - lam)
    if Q <= q_b:
        return 0.0
    if Q >= 1:
        return math.inf
    return math.log((1 - q_b) / (1 - Q)) / (-math.log(1 - q_o))


def illustrate_iso():
    if not STATS.exists():
        return None
    st = json.loads(STATS.read_text(encoding="utf-8"))
    try:
        lam = st["p0b"]["rho_onset"]["UAVDT"]["m0.00|w3"]["lambda_per_frame"]
        M_med = st["E0"]["UAVDT"]["main"]["M_median"]
    except KeyError:
        return None
    rows = []
    for Lm in (3, 5, 10, 30):
        T = Lm + 1
        for eps in (0.05, 0.01):
            for c in (0.0, 0.02, 0.05):
                Q = ((1 - eps) / T - lam - c) / (1 - lam)
                for q_b in (0.0, 0.05):
                    for q_o in (0.005, 0.01, 0.02):
                        mi = M_iso(T, eps, lam, q_b, q_o, c)
                        rows.append(dict(Lmax=Lm, T=T, eps=eps, c=c, lam=lam, a_periodic=(1 - eps) / T, q_out_max=Q,
                                         q_b=q_b, q_o=q_o, M_iso=(None if mi == math.inf else mi),
                                         gate_cheaper_at_M_median=bool(mi > M_med)))
    return dict(lambda_per_frame=lam, M_median=M_med, table=rows)


# ======================================================================= §6 P3
def q_out_eff(q_b, q_o, M):
    return 1 - (1 - q_b) * (1 - q_o) ** M


def can_win(T, w, R, rho, qo, r=1.0, c=0.0):
    qs = np.linspace(0, 1, 401)
    d = np.atleast_1d(delta(T, w, R, qs, qo, rho, r, c))
    return bool(d.max() > 1e-12), float(qs[int(np.argmax(d))]), float(d.max())


def M_star(T, w, R, rho, q_b, q_o, r=1.0, c=0.0, Mmax=400):
    """M* = M nhỏ nhất sao cho với mọi M' ≥ M không gate nào thắng (quét tới Mmax; kiểm bằng cận cần)."""
    wins = [can_win(T, w, R, rho, q_out_eff(q_b, q_o, M), r, c)[0] for M in range(Mmax + 1)]
    last = max([i for i, x in enumerate(wins) if x], default=-1)
    return last + 1


def M_nec(T, R, rho, q_b, q_o, c=0.0):
    """Cận trên theo điều kiện cần (N-G), r=1: q_out,eff ≥ (κ − cR/(R−1))/(1−ρ) ⇒ không thắng."""
    kap = (1 - T / R) / (T * (1 - 1 / R)) if R != math.inf else 1 / T
    cR = c * (R / (R - 1) if R != math.inf else 1.0)
    qmax = (kap - cR) / (1 - rho)
    if qmax <= q_b:
        return 0.0
    if qmax >= 1:
        return math.inf
    return math.log((1 - q_b) / (1 - qmax)) / (-math.log(1 - q_o))


def check_P3():
    # (1) q_out,eff MC
    zs = []
    for q_b, q_o, M in [(0.0, 0.01, 14), (0.05, 0.02, 7), (0.2, 0.005, 37), (0.0, 0.05, 3)]:
        n = 200_000
        fire = (rng.random(n) < q_b) | (rng.binomial(M, q_o, n) > 0)
        zs.append(z_of(fire.astype(float), q_out_eff(q_b, q_o, M)))
    record("q_out,eff = 1 − (1−q_b)(1−q_o)^M", "§6 Lemma P3.1", 4 * 200_000, zlist=zs)
    # (2) M* ≤ M_nec và với M ≥ ⌈M_nec⌉ không gate nào thắng (r=1)
    ok, n = True, 0
    for T, w, R in [(4, 3, 60), (6, 5, 30), (11, 5, math.inf)]:
        for rho in (0.05, 0.10, 0.16):
            for q_b in (0.0, 0.05):
                for q_o in (0.005, 0.02, 0.05):
                    ms = M_star(T, w, R, rho, q_b, q_o, Mmax=120)
                    mn = M_nec(T, R, rho, q_b, q_o)
                    ok &= (ms <= math.ceil(mn) if mn != math.inf else True)
                    n += 1
    record("M* ≤ M_nec (cận đóng) — r=1", "§6 Thm P3(i)", n, max_err=0.0, passed=ok)
    # (3) camera di động không bù: q_b → 1 ⇒ không thắng; có bù chi phí c: không thắng khi (1−ρ)q_out + cR/(R−1) ≥ κ
    no_win = all(not can_win(T, w, R, rho, q_out_eff(0.999, 0.01, 0))[0] for T, w, R in [(4, 3, 60), (6, 5, 30)] for rho in (0.05, 0.16))
    c_kill = all(not can_win(6, 5, 30, 0.10, 0.0, 1.0, c)[0] for c in (0.2, 0.3))
    record("q_b→1 ⇒ không gate thắng; chi phí cue c đủ lớn ⇒ không thắng", "§6 Thm P3(ii)", 6, max_err=0.0, passed=no_win and c_kill)


def illustrate_P3():
    """Minh hoạ số: ρ_onset(w) từ A5 (UAVDT), M trung vị UAVDT; q_b, q_o là giá trị MINH HOẠ (đo ở P2)."""
    if not STATS.exists():
        return None
    st = json.loads(STATS.read_text(encoding="utf-8"))
    try:
        rho = {w: st["p0b"]["rho_onset"]["UAVDT"][f"m0.00|w{w}"]["pooled"] for w in (3, 5, 10)}
        M_med = st["E0"]["UAVDT"]["main"]["M_median"]
    except KeyError:
        return None
    rows = []
    for Lm, w in ((3, 3), (5, 5), (10, 10)):
        T = Lm + 1
        for R in (math.inf, 60, 30):
            for q_b in (0.0, 0.05, 0.2):
                for q_o in (0.005, 0.01, 0.02, 0.05):
                    ms = M_star(T, w, R, rho[w], q_b, q_o, Mmax=200)
                    mn = M_nec(T, R, rho[w], q_b, q_o)
                    win_med = can_win(T, w, R, rho[w], q_out_eff(q_b, q_o, int(M_med)))
                    rows.append(dict(Lmax=Lm, T=T, w=w, R=("inf" if R == math.inf else R), rho=rho[w], q_b=q_b, q_o=q_o,
                                     M_star=ms, M_nec=(None if mn == math.inf else mn), M_median=M_med,
                                     gate_can_win_at_M_median=win_med[0], best_q_in=win_med[1], best_delta=win_med[2]))
    # q_in* tại vài ô (q_out cố định), r=1 và r=0.8
    thr = []
    for Lm, w in ((3, 3), (5, 5)):
        T = Lm + 1
        for R in (math.inf, 30):
            for qo in (0.0, 0.01, 0.03, 0.05):
                for r in (1.0, 0.8):
                    qs = np.linspace(0, 1, 1001)
                    d = np.atleast_1d(delta(T, w, R, qs, qo, rho[w], r))
                    win = d > 1e-12
                    thr.append(dict(Lmax=Lm, w=w, R=("inf" if R == math.inf else R), q_out=qo, r=r, rho=rho[w],
                                    q_in_star=(float(qs[np.argmax(win)]) if win.any() else None),
                                    q_in_upper=(float(qs[len(qs) - 1 - np.argmax(win[::-1])]) if win.any() else None),
                                    max_delta=float(d.max())))
    return dict(rho_onset=rho, M_median=M_med, M_star_table=rows, q_in_star_table=thr,
                note="q_b, q_o, c là giá trị minh hoạ; giá trị thật đo trên TRAIN ở P2 trước khi đóng băng PREREG")


if __name__ == "__main__":
    check_lemma1()
    check_censoring()
    check_P1()
    p2 = check_P2()
    check_gate()
    check_isoKPI()
    check_multilook()
    check_P3()
    ill = illustrate_P3()
    iso = illustrate_iso()
    out = dict(seed=SEED, all_passed=all(c["passed"] for c in CHECKS), n_checks=len(CHECKS), checks=CHECKS, P2_example=p2, P3_illustration=ill, isoKPI_illustration=iso)
    (OUT / "theory_checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("ALL PASSED" if out["all_passed"] else "SOME FAILED", len(CHECKS))
