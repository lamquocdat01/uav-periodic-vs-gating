"""Unit test cho công thức phơi nhiễm (tự kiểm P0). Ghi kết quả vào results/p0/unit_tests.json.

T1: r=1 ⇒ miss = (1 − D/S)_+ khớp tới 1e-12 trên ≥100 cặp (D,S).
T2: Monte Carlo 10^5 mẫu (seed=42): pha ngẫu nhiên đều, đếm N số khung detector rơi vào sự kiện;
    E[N], P(N=0), E[0.5^N] khớp công thức trong 3σ.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from p0_stats import expected_N, miss_r, p_zero  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "p0" / "unit_tests.json"


def t1():
    pairs = [(D, S) for D in range(1, 61) for S in (1, 2, 3, 4, 5, 7, 10, 12, 20, 24, 40, 60, 96, 240)]
    D = np.array([p[0] for p in pairs])
    S = np.array([p[1] for p in pairs])
    got = np.array([miss_r(d, s, 1.0) for d, s in pairs], dtype=float)
    want = np.clip(1 - D / S, 0, None)
    err = float(np.max(np.abs(got - want)))
    pz = float(np.max(np.abs(np.array([p_zero(d, s) for d, s in pairs], dtype=float) - want)))
    return dict(n_pairs=len(pairs), max_abs_err=err, max_abs_err_pzero=pz, passed=bool(err <= 1e-12 and pz <= 1e-12))


def t2():
    rng = np.random.default_rng(42)
    n_mc = 100_000
    cases = [(1, 5), (3, 10), (7, 5), (12, 10), (25, 12), (40, 60), (100, 24), (8, 8), (13, 3), (2, 240)]
    res, ok = [], True
    for D, S in cases:
        phase = rng.integers(0, S, size=n_mc)          # detector ở các khung t ≡ phase (mod S)
        t0 = 0                                          # sự kiện chiếm [t0, t0+D-1]
        first = (phase - t0) % S                        # khung detector đầu tiên ≥ t0 (tương đối)
        N = np.where(first < D, (D - 1 - first) // S + 1, 0)
        for name, sample, want in (("E[N]", N, float(expected_N(D, S))),
                                   ("P(N=0)", (N == 0).astype(float), float(p_zero(D, S))),
                                   ("E[0.5^N]", 0.5 ** N, float(miss_r(D, S, 0.5))),
                                   ("E[0.2^N]", 0.2 ** N, float(miss_r(D, S, 0.8)))):
            m, sd = float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n_mc))
            dev = abs(m - want)
            passed = dev <= 3 * sd + 1e-15
            ok &= passed
            res.append(dict(D=D, S=S, stat=name, mc=m, formula=want, se=sd, z=(dev / sd) if sd > 0 else 0.0, passed=bool(passed)))
    return dict(n_mc=n_mc, seed=42, cases=res, passed=bool(ok), max_z=float(max(r["z"] for r in res)))


if __name__ == "__main__":
    out = {"T1_r1_closed_form": t1(), "T2_monte_carlo": t2()}
    out["all_passed"] = out["T1_r1_closed_form"]["passed"] and out["T2_monte_carlo"]["passed"]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("T1", out["T1_r1_closed_form"])
    print("T2 passed", out["T2_monte_carlo"]["passed"], "max_z", out["T2_monte_carlo"]["max_z"])
    sys.exit(0 if out["all_passed"] else 1)
