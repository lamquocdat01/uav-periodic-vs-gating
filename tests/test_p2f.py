"""P2F test (tổng hợp): luật H8 (r_main − r_low ≥ 0,05), luật khung M0207 (≤ khung GT cuối) + biên theo bề rộng thật (M0901),
audit khung trước onset A1–A4 (P2G, chỉ mô tả) + luật không loại, luật chi phí."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "p2"))
from p0_parse import gt_frame_range, is_border  # noqa: E402
from p2_audit_prebirth import audit_event, class_table, classify  # noqa: E402
from p2_prereg_build import H8_DELTA, h8_rule, select_operating_points  # noqa: E402


def test_h8_rule_threshold():
    r = {"s1024": 0.944, "n1024": 0.921, "s640": 0.931, "n640": 0.890}
    out = h8_rule(r, "s1024", candidates=["n1024", "s640", "n640"])
    assert H8_DELTA == 0.05
    assert out["valid"] == ["n640"] and out["pairs"] == [["s1024", "n640"]] and out["status"] == "pairs"
    # đúng biên: gap = 0,05 hợp lệ; 0,0499 không
    out = h8_rule({"m": 0.95, "a": 0.90, "b": 0.9001}, "m", candidates=["a", "b"])
    assert out["valid"] == ["a"]
    # không mức nào hợp lệ → none_valid (H8 thành mô tả); còn ứng viên chưa đo → pending
    assert h8_rule({"m": 0.95, "a": 0.93}, "m", candidates=["a"])["status"] == "none_valid"
    assert h8_rule({"m": 0.95, "a": 0.93}, "m", candidates=["a", "z"])["status"] == "pending"
    # tối đa 2 cặp, mức r thấp nhất trước
    out = h8_rule({"m": 0.95, "a": 0.80, "b": 0.85, "c": 0.70}, "m", candidates=["a", "b", "c"])
    assert out["pairs"] == [["m", "c"], ["m", "a"]]


def test_m0207_frame_rule():
    assert gt_frame_range(885, 571) == 571     # M0207: ảnh dư sau khung GT cuối bị bỏ
    assert gt_frame_range(1906, 1906) == 1906
    assert gt_frame_range(0, 571) == 571       # chưa có frames → phạm vi GT
    assert gt_frame_range(500, 571) == 500     # thiếu ảnh → chỉ khung có ảnh


def test_border_uses_real_width():
    # M0901: box có mép phải x+w = 950 → biên khi W = 960 (dải 2 % = 19,2 px), không biên khi giả định W = 1024
    assert is_border(900, 200, 50, 30, 960, 540) and not is_border(900, 200, 50, 30, 1024, 540)


def _dets(frames, box, score=0.9, jitter=1.0):
    return pd.DataFrame([dict(frame=f, x=box[0] + jitter, y=box[1], w=box[2], h=box[3], score=score) for f in frames],
                        columns=["frame", "x", "y", "w", "h", "score"])


NOPRIOR = pd.DataFrame(columns=["occlusion", "out_of_view"])


def _prior(occ, oov):
    return pd.DataFrame(dict(occlusion=occ, out_of_view=oov))


def test_audit_match_counting():
    box, b = (100.0, 100.0, 40.0, 30.0), 50
    ev = lambda d, **k: audit_event(box, d, b, NOPRIOR, "interior", 1024, 540, **k)  # noqa: E731
    assert ev(_dets(range(b - 5, b), box))["n_match"] == 5
    assert ev(_dets([b - 2, b - 1], box))["label"] == "true-birth"                       # 2/5 → true-birth
    assert ev(_dets(range(b - 5, b), box, score=0.2))["n_match"] == 0                    # score < 0,25 không tính
    assert ev(_dets(range(b - 5, b), box, jitter=22.0))["n_match"] == 0                  # IoU 0,29 < 0,3
    assert ev(_dets(range(b - 5, b), box, jitter=21.0))["n_match"] == 5                  # IoU 0,31
    r = ev(_dets([b - 4, b - 2, b - 1], box))
    assert r["onset_alt"] == b - 4 and r["shift"] == 4                                    # khung khớp sớm nhất
    assert audit_event(box, _dets(range(1, 4), box), 4, NOPRIOR, "interior", 1024, 540)["label"] == "censored-start"


def test_audit_classes_A1_A4():
    """≥ 3/5 khớp: A1 partial-entry / A2 visibility-transition / A3 annotation-late / A4 true-birth; ≤ 2/5 → true-birth."""
    assert classify(5, False, "border", 1.0, NOPRIOR) == "partial-entry"                 # chạm mép ≤ 2 px
    assert classify(5, False, "border", 30.0, _prior([1], [2])) == "partial-entry"       # GT trước đó out_of_view = 2
    assert classify(5, False, "border", 30.0, _prior([2], [1])) == "visibility-transition"
    assert classify(5, False, "interior", 30.0, _prior([1], [2])) == "visibility-transition"
    assert classify(5, False, "interior", 30.0, _prior([2, 1], [1, 1])) == "visibility-transition"
    assert classify(4, False, "interior", 30.0, NOPRIOR) == "annotation-late"            # ID mới, detector thấy ≥ 3/5
    assert classify(3, False, "border", 3.0, NOPRIOR) == "annotation-late"               # border nhưng cách mép > 2 px
    assert classify(5, False, "interior", 30.0, _prior([1], [1])) == "true-birth"        # còn lại (A4)
    assert classify(2, False, "interior", 30.0, NOPRIOR) == "true-birth"
    assert classify(5, True, "border", 0.0, NOPRIOR) == "censored-start"
    au = pd.DataFrame(dict(e3_type=["border", "border", "interior"], label=["partial-entry", "true-birth", "annotation-late"]))
    t = class_table(au)
    assert t["border"]["partial-entry"] == 1 and t["interior"]["share_annotation_late"] == 1.0 and t["all"]["n"] == 3


def test_no_exclusion_policy():
    """P2G-Q5: audit chỉ mô tả — p0_events không còn bước lọc; tập E3 TRAIN (VISIBLE chính) = đúng tập mà audit xét."""
    import p0_events
    import p2_audit_prebirth
    assert p2_audit_prebirth.POLICY == "descriptive" and not hasattr(p0_events, "filter_prebirth")
    P0 = ROOT / "results" / "p0"
    a, e = P0 / "audit_prebirth_train.json", P0 / "events_E3_uavdt.parquet"
    if a.exists() and e.exists():
        n_audit = json.loads(a.read_text(encoding="utf-8"))["n_events"]
        e3 = pd.read_parquet(e)
        n = int(((e3.split == "train") & (e3.vis_def == "main") & (~e3.never_visible)).sum())
        assert n == n_audit, (n, n_audit)
        assert "excluded" not in json.loads(a.read_text(encoding="utf-8"))



def test_cost_rule_excludes_expensive_cue():
    ch = dict(rho_onset={"5": 0.15}, operating_points=[
        dict(cue="cheap", w=5, theta=0.1, q_in=0.3, q_out=0.05, c=0.02, q_b={"all": 0.05}, q_o=None),
        dict(cue="dear", w=5, theta=0.1, q_in=0.9, q_out=0.0, c=1.02, q_b={"all": 0.0}, q_o=None)])
    ops, log = select_operating_points(ch)
    assert [o["cue"] for o in ops] == ["cheap"]
    assert next(x for x in log if x["cue"] == "dear")["excluded"]


def test_r_dagger_closed_form_and_family():
    """P2H-Q7: r† số (builder) khớp công thức đóng Remark G2′(iv) khi R = ∞, q_out = 0, w' = T, A-sparse:
    r† = (1 − (a/q_in)^{1/(T−1)})/q_in; ô H8 có r ≤ r† → "—" và ra khỏi họ."""
    import math
    from p2_prereg_build import apply_family_rules, r_dagger
    D = np.full(50, 10)                                  # T = min(10, L+1) = 4 với L = 3
    for q_in, rho in ((1.0, 0.09), (0.8, 0.1)):
        a = rho * q_in                                   # a_G = ρ q_in khi R = ∞, q_out = 0, c = 0
        want = min(1.0, (1 - (a / q_in) ** (1 / 3)) / q_in)
        got = r_dagger(D, 3, 5, math.inf, q_in, 0.0, 0.0, rho, mode="asparse")
        assert abs(got - want) < 2e-4, (got, want)
    base = dict(eligible=True, comparison="r_pair", pool=None, L=3, kpi="H8_L3")
    cells = [dict(base, rdag_ok=True), dict(base, rdag_ok=False), dict(base, comparison="matched_cost", kpi="3")]
    fam = apply_family_rules(cells, fmax=80)
    assert [c["confirmatory"] for c in cells] == [True, False, True] and fam["log"][-1]["step"] == "H8_rdagger" and fam["n_final"] == 2


def test_edge_sensitivity_5px():
    """P2H-Q9: border cách mép 3–5 px: annotation-late ở ngưỡng 2 px, partial-entry ở ngưỡng 5 px (chỉ mô tả)."""
    assert classify(5, False, "border", 4.0, NOPRIOR) == "annotation-late"
    assert classify(5, False, "border", 4.0, NOPRIOR, edge_max=5) == "partial-entry"
    assert classify(5, False, "interior", 4.0, NOPRIOR, edge_max=5) == "annotation-late"   # interior không đổi


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
