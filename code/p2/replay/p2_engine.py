"""P2A-A3/A4/A5: bảng sự kiện KPI, chính sách, replay offline trên hit table.

Sự kiện KPI (VISIBLE chính, CỬA SỔ SINH W=120 như P0b; có cờ để lấy mọi sự kiện):
  E2       đoạn visible của track xe
  E3       xe mới (sinh sau khung đầu), đoạn visible đầu
  E3ROI<m> E3 trong ROI biên m (m ∈ {0, 0.05}): đoạn liên tiếp đầu tiên tâm GT trong ROI
Một sự kiện "được phát hiện" nếu có khung trong cửa sổ KPI [start, start+D) mà detector CHẠY và box GT của track
được ghép (hit). Onset latency L = (khung chạy+hit đầu tiên) − start; P(L ≤ Lmax).

Chính sách → ma trận run (P × G) bool trên trục khung toàn cục G (nối các chuỗi), P = số pha:
  periodic(S)            lattice ⌊φ + kS⌋ theo từng chuỗi, S thực ≥ 1; φ: S nguyên ≤ max_exact → mọi pha nguyên
                         (trung bình chính xác), còn lại n_phase pha đều (j+½)S/n
  gate_or_refresh        run = fire ∨ (t ≡ φ mod R); R=∞ → gate thuần (P=1)
  onset_oracle           fire đúng khung onset mọi E3 + Bernoulli(q_out) ở khung khác
  oracle_per_seq_stride  trần: S riêng từng chuỗi (lưới) tối ưu tổng miss ở cùng ngân sách (Lagrange)
Chi phí: a = (#khung chạy detector)/(#khung) + c·(#khung tính cue)/(#khung).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
P0 = ROOT / "results" / "p0"
W_BIRTH = 120
BIG = np.int32(1 << 30)


# ---------------------------------------------------------------- khung
class Timeline:
    """Trục khung toàn cục: chuỗi seqs (thứ tự cố định), mỗi chuỗi khung 1..L_s."""

    def __init__(self, n_frames: dict):
        self.seqs = list(n_frames)
        self.L = np.array([n_frames[s] for s in self.seqs], np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.L)[:-1]])
        self.G = int(self.L.sum())
        self.idx = {s: i for i, s in enumerate(self.seqs)}
        self.seq_of_g = np.repeat(np.arange(len(self.seqs)), self.L)

    def g(self, seq, frame):
        return self.off[self.idx[seq]] + np.asarray(frame) - 1


# ---------------------------------------------------------------- sự kiện
def load_events(split="train", birth_only=True, dataset="uavdt"):
    """dict tên → DataFrame(seq, track_id, start, D, + cột phân tầng)."""
    fr = pd.read_parquet(P0 / f"frames_{dataset}.parquet")
    last = fr.groupby("seq").frame.max()
    e3 = pd.read_parquet(P0 / f"events_E3_{dataset}.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main")]
    e2 = pd.read_parquet(P0 / f"events_E2_{dataset}.parquet")
    e2 = e2[(e2.vis_def == "main") & (e2.group == "VEHICLE")]
    roi = pd.read_parquet(P0 / f"roi_events_{dataset}.parquet")
    roi = roi[roi.enters_roi]
    strata_cols = ["M_at_birth", "sqrt_area_median", "speed_px_median", "altitude", "e3_type"]
    out = {}

    def win(d):
        d = d.copy()
        d["seq_last"] = d.seq.map(last)
        d["birth_window"] = (~d.left_censored.astype(bool)) & (d.start <= d.seq_last - W_BIRTH)
        return d[d.birth_window] if birth_only else d

    for name, d in (("E2", e2), ("E3", e3)):
        d = d[d.split == split] if split != "all" else d
        out[name] = win(d)
    base = e3.set_index(["seq", "track_id"])[strata_cols + ["split"]]
    for m in (0.0, 0.05):
        d = roi[roi.m == m].join(base, on=["seq", "track_id"], rsuffix="_e3")
        d = d[d.split == split] if split != "all" else d
        out[f"E3ROI{int(round(m * 100))}"] = win(d)
    for name, d in out.items():
        d = d.reset_index(drop=True)
        if "M_at_birth" not in d.columns:
            d["M_at_birth"] = d.get("M_at_start")
        d["M_bin"] = pd.cut(d.M_at_birth, [-np.inf, 5.5, 20.5, np.inf], labels=["1-5", "6-20", ">20"]).astype(str)
        d["size_bin"] = pd.cut(d.sqrt_area_median, [-np.inf, 16, 32, 64, np.inf], right=False, labels=["<16", "16-32", "32-64", ">=64"]).astype(str)
        sp = d.speed_px_median
        d["speed_tertile"] = pd.qcut(sp.rank(method="first"), 3, labels=["slow", "mid", "fast"]).astype(str) if sp.notna().sum() >= 3 else "unknown"
        d["ego"] = "unknown"  # chưa có frames → chưa có hovering/moving
        out[name] = d
    return out


def onset_frames(split="train", dataset="uavdt"):
    """Khung onset (visible đầu) của MỌI sự kiện E3 (không lọc cửa sổ) — dùng cho cue giả lập/onset oracle."""
    e3 = pd.read_parquet(P0 / f"events_E3_{dataset}.parquet")
    e3 = e3[(~e3.never_visible) & (e3.vis_def == "main")]
    return e3[e3.split == split][["seq", "start"]] if split != "all" else e3[["seq", "start"]]


# ---------------------------------------------------------------- bảng hit phẳng theo sự kiện
class EventSet:
    """Các mục hit trong cửa sổ KPI, sắp theo sự kiện rồi theo rel = frame − start."""

    def __init__(self, ev: pd.DataFrame, hits: dict, tl: Timeline):
        self.ev = ev.reset_index(drop=True)
        self.tl = tl
        self.n = len(self.ev)
        self.seq_idx = np.array([tl.idx[s] for s in self.ev.seq], np.int64)
        gs, rels, eids = [], [], []
        by_seq = {s: h for s, h in hits.items()}
        for i, e in enumerate(self.ev.itertuples(index=False)):
            h = by_seq.get(e.seq)
            if h is None or not len(h):
                continue
            f = h.frame.values[(h.tid.values == e.track_id) & (h.frame.values >= e.start) & (h.frame.values < e.start + e.D)]
            if len(f):
                f = np.sort(f)
                gs.append(tl.g(e.seq, f))
                rels.append(f - e.start)
                eids.append(np.full(len(f), i))
        self.g = np.concatenate(gs) if gs else np.array([], np.int64)
        self.rel = (np.concatenate(rels) if rels else np.array([], np.int64)).astype(np.int32)
        self.eid = np.concatenate(eids) if eids else np.array([], np.int64)
        self.has = np.zeros(self.n, bool)
        self.has[np.unique(self.eid)] = True
        self.seg_start = np.searchsorted(self.eid, np.flatnonzero(self.has))

    @classmethod
    def from_hitmask(cls, ev, hitmask_by_seq, tl):
        """Tiện cho test: hitmask_by_seq[seq] = DataFrame(tid, frame)."""
        return cls(ev, hitmask_by_seq, tl)

    def first_latency(self, run):
        """run: (P × G) bool → (P × n) int32 độ trễ onset nhỏ nhất (BIG = không phát hiện)."""
        run = np.atleast_2d(run)
        out = np.full((run.shape[0], self.n), BIG, np.int32)
        if len(self.g) == 0:
            return out
        vals = np.where(run[:, self.g], self.rel[None, :], BIG)
        out[:, self.has] = np.minimum.reduceat(vals, self.seg_start, axis=1)
        return out


# ---------------------------------------------------------------- chính sách
def phases_for(S, n_phase=20, max_exact=120):
    if abs(S - round(S)) < 1e-9 and round(S) <= max_exact:
        return np.arange(int(round(S)), dtype=float)
    return (np.arange(n_phase) + 0.5) * S / n_phase


def periodic(tl: Timeline, S, n_phase=20, max_exact=120):
    """Ma trận run (P × G) cho lịch tuần hoàn stride thực S ≥ 1 (lattice ⌊φ + kS⌋ trên mỗi chuỗi)."""
    ph = phases_for(S, n_phase, max_exact)
    run = np.zeros((len(ph), tl.G), bool)
    for s, (L, off) in enumerate(zip(tl.L, tl.off)):
        k = np.arange(int(math.ceil(L / S)) + 1)
        for p, phi in enumerate(ph):
            t = np.floor(phi + k * S + 1e-9).astype(np.int64)
            t = t[(t >= 0) & (t < L)]
            run[p, off + t] = True
    return run


def refresh(tl: Timeline, R, n_phase=20):
    if R is None or R == math.inf:
        return np.zeros((1, tl.G), bool)
    R = int(R)
    ph = np.arange(R) if R <= n_phase else np.unique(np.floor((np.arange(n_phase) + 0.5) * R / n_phase).astype(int))
    run = np.zeros((len(ph), tl.G), bool)
    local = np.arange(tl.G) - np.repeat(tl.off, tl.L)
    for p, phi in enumerate(ph):
        run[p] = (local % R) == phi
    return run


def gate_or_refresh(fire, tl: Timeline, R=math.inf, n_phase=20):
    """fire: (G,) bool (cue ≥ θ). run = fire ∨ refresh(R)."""
    return refresh(tl, R, n_phase) | np.asarray(fire, bool)[None, :]


def cue_fire(score, theta):
    return np.asarray(score) >= theta


def onset_mask(tl: Timeline, onsets: pd.DataFrame, w=1):
    """(G,) bool: khung nằm trong ≥1 cửa sổ [onset, onset+w) của E3 (cắt ở cuối chuỗi)."""
    m = np.zeros(tl.G, bool)
    for seq, g in onsets.groupby("seq"):
        if seq not in tl.idx:
            continue
        L, off = tl.L[tl.idx[seq]], tl.off[tl.idx[seq]]
        for s in g.start.values:
            a = int(s) - 1
            m[off + a: off + min(a + w, L)] = True
    return m


def sim_cue(tl: Timeline, in_mask, q_in, q_out, seed, q_out_frame=None):
    """Cue giả lập theo A-gate: Bernoulli(q_in) trong cửa sổ khởi phát, Bernoulli(q_out) (hoặc q_out_frame[t]) ngoài."""
    rng = np.random.default_rng(seed)
    u = rng.random(tl.G)
    qo = q_out if q_out_frame is None else q_out_frame
    return np.where(in_mask, u < q_in, u < qo)


def onset_oracle(tl, onsets, q_out, seed):
    """Chạy đúng khung onset của mọi E3 + Bernoulli(q_out) ở mọi khung khác (không refresh)."""
    om = onset_mask(tl, onsets, 1)
    return (om | (np.random.default_rng(seed).random(tl.G) < q_out))[None, :]


def activation(run, c=0.0, cue_frac=1.0):
    """Chi phí mỗi pha: (#chạy)/(#khung) + c·(tỉ lệ khung tính cue)."""
    return np.atleast_2d(run).mean(axis=1) + c * cue_frac


# ---------------------------------------------------------------- KPI
def kpis(lat, Lmax_list=(3, 5, 10, 30)):
    """lat (P × n) → dict: det_prob (n,) = P_phase(phát hiện), lat_le[Lmax] (n,)."""
    out = {"det": (lat < BIG).mean(axis=0)}
    for Lm in Lmax_list:
        out[f"L{Lm}"] = (lat <= Lm).mean(axis=0)
    return out


def matched_periodic_S(tl, a_target, c_gate=0.0, tol=0.002, n_phase=20, lo=1.0, hi=None, iters=40):
    """Bisection trên S thực để activation periodic (không cue) = a_target (chi phí gate gồm cả cue). Trả (S, a_P)."""
    if a_target >= 1:
        return 1.0, 1.0
    hi = hi or max(2.0, 4.0 / max(a_target, 1e-4))
    S = 1.0 / a_target
    for _ in range(iters):
        a = activation(periodic(tl, S, n_phase)).mean()
        if abs(a - a_target) < tol / 4:
            break
        if a > a_target:
            lo = S
        else:
            hi = S
        S = 0.5 * (lo + hi)
    return S, float(activation(periodic(tl, S, n_phase)).mean())


def periodic_iso(tl, es, target, kind="miss", Lmax=None, n_phase=20, S_max=400.0, iters=30):
    """S lớn nhất (bisection, giả định đơn điệu) sao cho KPI đạt: miss ≤ target (kind='miss') hoặc
    P(L ≤ Lmax) ≥ 1 − target (kind='lat'). Trả (S, a_P, kpi_at_S)."""
    def ok(S):
        k = kpis(es.first_latency(periodic(tl, S, n_phase)), (Lmax,) if Lmax else ())
        v = 1 - k["det"].mean() if kind == "miss" else 1 - k[f"L{Lmax}"].mean()
        return v <= target + 1e-12, v
    good, v1 = ok(1.0)
    if not good:
        return None, None, v1
    lo, hi = 1.0, S_max
    if ok(hi)[0]:
        lo = hi
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if ok(mid)[0]:
                lo = mid
            else:
                hi = mid
    return lo, float(activation(periodic(tl, lo, n_phase)).mean()), ok(lo)[1]


def oracle_per_seq_stride(tl, es, a_target, S_grid=None, n_phase=20):
    """Trần: mỗi chuỗi chọn S_s trong lưới để cực tiểu Σ miss với Σ_s L_s/S_s ≤ a_target·G (Lagrange + bisection λ)."""
    S_grid = S_grid if S_grid is not None else np.unique(np.r_[np.arange(1, 21), np.arange(22, 61, 2), np.arange(65, 241, 5)]).astype(float)
    ns = len(tl.seqs)
    miss_cnt = np.zeros((ns, len(S_grid)))
    act = np.zeros((ns, len(S_grid)))
    for j, S in enumerate(S_grid):
        run = periodic(tl, S, n_phase)
        det = kpis(es.first_latency(run))["det"]
        miss_cnt[:, j] = np.bincount(es.seq_idx, weights=1 - det, minlength=ns)
        act[:, j] = np.add.reduceat(run.mean(axis=0), tl.off)  # số khung chạy kỳ vọng mỗi chuỗi
    budget = a_target * tl.G
    lo, hi = 0.0, 1e4
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        pick = np.argmin(miss_cnt + lam * act, axis=1)
        if act[np.arange(ns), pick].sum() > budget:
            lo = lam
        else:
            hi = lam
    pick = np.argmin(miss_cnt + hi * act, axis=1)
    return dict(S_per_seq=S_grid[pick].tolist(), a=float(act[np.arange(ns), pick].sum() / tl.G),
                miss=float(miss_cnt[np.arange(ns), pick].sum() / max(es.n, 1)),
                miss_per_seq_counts=miss_cnt[np.arange(ns), pick].tolist())
