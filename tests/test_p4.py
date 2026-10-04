"""P4 tests: the necessary condition used in the design rule (Sec. VII) and the generated tables use macros only.

Claim (derived from Prop. G1, G2 and Lemma 1; R = inf, A-gate, A-sparse, A-ind, T(a_G + c) <= 1, any r in (0, 1]):
  Delta = miss_P - miss_G <= r [ J (w' - T rho) - T c ],  J = q_in - q_out,
so J (w' - T rho) <= T c  (i.e. J <= J* = T c / (w' - T rho) when w' > T rho)  =>  Delta <= 0.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from theory_checks import a_gate, lemma_miss, miss_gate  # noqa: E402


def test_first_order_bound_is_upper_bound():
    rng = np.random.default_rng(7)
    n_ok = 0
    for _ in range(20000):
        T = int(rng.integers(2, 12))
        w = int(rng.integers(1, 11))
        wp = min(w, T)
        rho = float(rng.uniform(0, 0.4))
        q_out = float(rng.uniform(0, 0.3))
        q_in = float(rng.uniform(q_out, 1))
        c = float(rng.uniform(0, 0.2))
        r = float(rng.uniform(0.05, 1))
        a = a_gate(rho, q_in, q_out, math.inf) + c
        if T * a > 1:
            continue
        delta = float(lemma_miss(np.array([T]), 1.0 / a, r)[0]) - miss_gate(T, w, math.inf, q_in, q_out, r)
        bound = r * ((q_in - q_out) * (wp - T * rho) - T * c)
        assert delta <= bound + 1e-12, (T, w, rho, q_in, q_out, c, r, delta, bound)
        n_ok += 1
    assert n_ok > 1000, n_ok


def test_tables_contain_no_literal_numbers():
    for p in (ROOT / "paper" / "tables").glob("tab_*.tex"):
        body = "\n".join(l.split("%")[0] for l in p.read_text(encoding="utf-8").splitlines())
        body = re.sub(r"\\(begin|end|label|ref|cmidrule|multicolumn|setlength)\{[^}]*\}(\{[^}]*\})?", "", body)
        body = re.sub(r"\\cmidrule\([a-z]*\)\{[^}]*\}", "", body)
        body = re.sub(r"\\[A-Za-z]+", "", body)
        body = re.sub(r"\$[^$]*\$", "", body)
        assert not re.search(r"(?<![A-Za-z0-9@])\d", body), (p.name, re.findall(r".{10}(?<![A-Za-z0-9@])\d.{10}", body)[:3])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
