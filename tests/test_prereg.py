"""P2A-C test (tổng hợp): builder từ chối kênh không phải TRAIN; sinh ô với dấu dự đoán đúng theo công thức;
scorer áp đúng luật dấu/Holm; Holm và paired bootstrap của p2_stats."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "p2"))
sys.path.insert(0, str(ROOT / "code" / "p2" / "replay"))
from p2_prereg_build import build, theta_star  # noqa: E402
from p2_prereg_score import score  # noqa: E402
from p2_stats import boot_weights, holm, paired_delta  # noqa: E402


def fake_events():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(dict(seq=[f"S{i % 10}" for i in range(n)], D=rng.integers(1, 400, n),
                             M_bin=np.where(np.arange(n) % 2 == 0, "6-20", ">20")))


def channel(split="train"):
    return dict(split=split, simulated=True, lambda_per_frame=0.04, rho_onset={"1": 0.04, "3": 0.1, "5": 0.16},
                operating_points=[dict(cue="good", w=5, q_in=0.9, q_out=0.0, q_b={"all": 0.0}, q_o=None, c=0.0),
                                  dict(cue="noisy", w=5, q_in=0.9, q_out=0.6, q_b={"all": 0.0}, q_o=None, c=0.0)],
                r_levels={"r1": 1.0})


def test_builder_requires_train():
    try:
        build(channel("test"), fake_events())
    except AssertionError:
        return
    raise AssertionError("builder chấp nhận kênh TEST")


def test_builder_signs():
    cells = pd.DataFrame(build(channel(), fake_events(), select=False))   # select=False: kiểm công thức trên cả 2 operating point
    lat = cells[(cells.kpi == "3") & (cells.comparison == "matched_cost") & (cells.R == "inf")]
    assert (lat[lat.cue == "good"].sign_pred == "+").all()      # cue sạch, cửa sổ ngắn → gate thắng
    assert (lat[lat.cue == "noisy"].sign_pred == "<=0").all()   # q_out = 0.6 → không thắng
    # luật chốt 27/09: Lmax = 30, KPI miss và ε = 1% chỉ mô tả; họ xác nhận = latency 3/5/10, ε ∈ {2%, 5%}, H8
    assert not cells[cells.kpi.isin(["iso_L30", "miss", "30"])].confirmatory.any()
    assert not cells[(cells.comparison == "iso_kpi") & (cells.eps == 0.01)].confirmatory.any()
    mc3 = cells[(cells.kpi == "3") & (cells.comparison == "matched_cost")]
    assert mc3[mc3.R == 30].confirmatory.all() and not mc3[mc3.R == "inf"].confirmatory.any()   # P2C: họ xác nhận chỉ R = 30
    assert cells.confirmatory.any()


def test_theta_star_rule():
    """P2C-B1: Youden ở w=5 với a_G(R=30) ≤ 0,25; hoà → a_G nhỏ hơn; không điểm nào thoả → None."""
    rows = [dict(theta=1, q_in=0.9, q_out=0.10, c=0.0),    # J = 0.80 nhưng a_G ≈ 0.254 > 0.25 → loại
            dict(theta=2, q_in=0.72, q_out=0.04, c=0.0),   # J = 0.68, a_G ≈ 0.177
            dict(theta=3, q_in=0.70, q_out=0.02, c=0.0)]   # J = 0.68 (hoà), a_G ≈ 0.158 → chọn
    s = theta_star(rows, 0.16)
    assert s["theta"] == 3 and abs(s["J_star"] - 0.68) < 1e-9 and s["a_G_star"] <= 0.25
    assert theta_star([dict(theta=1, q_in=0.9, q_out=0.3, c=0.0)], 0.16) is None
    assert theta_star([dict(theta=1, q_in=0.5, q_out=0.0, c=0.3)], 0.16) is None   # c tính vào a_G


def test_family_reduction():
    """P2C-B2: 4 cue × 2 M_bin × 2 ego × 3 mức detector → thu gọn theo thứ tự cố định tới ≤ 80 ô."""
    rows = []
    for k in range(12):
        for i in range(20):
            rows.append(dict(seq=f"S{k}", D=5 + (7 * i + k) % 300, M_bin=("6-20" if i % 2 else ">20"), M_at_birth=(10 if i % 2 else 30)))
    ev = pd.DataFrame(rows)
    ego = {f"S{k}": ("hovering" if k < 6 else "moving") for k in range(12)}
    ops = [dict(cue=c, theta=0.5, w=5, q_in=0.8, q_out=0.02, q_b={"hovering": 0.01, "moving": 0.05}, q_o=0.001, c=0.02)
           for c in ("raw_diff", "ego_comp_diff", "border_band", "tiny_det")]
    ch = dict(split="train", rho_onset={"5": 0.16}, operating_points=ops,
              r_levels={"yolo26s_1024": 0.9, "yolo26n_1024": 0.8, "yolo26s_640": 0.7}, main_level="yolo26s_1024")
    info = {}
    cells = pd.DataFrame(build(ch, ev, ego_by_seq=ego, info=info))
    fam = info["family"]
    assert [s["step"] for s in fam["log"]] == ["start", "drop_eps5", "drop_L10", "H8_pool_M", "H8_L3", "H8_rdagger"], fam["log"]
    assert [s["n"] for s in fam["log"]] == [240, 192, 128, 96, 80, 80] and not fam["over_limit"]   # r† < mọi r ở kênh giả này
    conf = cells[cells.confirmatory]
    assert len(conf) == 80 and set(conf.R) == {30} and not (conf.L == 10).any()
    assert set(conf[conf.comparison == "iso_kpi"].eps) == {0.02}
    h8 = conf[conf.hypothesis == "H8"]
    assert set(h8.M_bin) == {"all"} and set(h8.L) == {3} and set(h8.r_level) == {"yolo26s_1024>yolo26n_1024", "yolo26s_1024>yolo26s_640"}
    assert set(conf[conf.comparison != "r_pair"].r_level) == {"yolo26s_1024"}
    # P2F-Q2: chênh r < 0,05 ở mọi mức → H8 ra khỏi họ xác nhận (mô tả), H2 + H3/H5 thu gọn theo cùng thứ tự
    ch2 = dict(ch, r_levels={"yolo26s_1024": 0.944, "yolo26n_1024": 0.921, "yolo26s_640": 0.931, "yolo26n_640": 0.91})
    info2 = {}
    c2 = pd.DataFrame(build(ch2, ev, ego_by_seq=ego, info=info2))
    assert info2["h8_rule"]["status"] == "none_valid" and info2["h8_pairs"] == []
    assert not c2[c2.confirmatory & (c2.hypothesis == "H8")].shape[0] and (c2.hypothesis == "H8").any()


def test_scorer_rules():
    """P2H-Q6: chấm 3 mức. "+": đúng iff lo > 0, sai iff hi ≤ 0; "<=0": đúng iff hi < δ_min, sai iff lo > δ_min; còn lại chưa kết luận."""
    from p2_prereg_build import MARGIN
    from p2_prereg_score import DELTA_MIN, judge, judge_iso
    assert DELTA_MIN == MARGIN == 0.005
    assert judge("+", 0.01, 0.2) == "correct" and judge("+", -0.1, 0.0) == "wrong" and judge("+", -0.02, 0.04) == "inconclusive"
    assert judge("<=0", -0.1, 0.004) == "correct" and judge("<=0", 0.006, 0.1) == "wrong" and judge("<=0", -0.01, 0.01) == "inconclusive"
    assert judge("<=0", 0.001, 0.0049) == "correct"            # CI dương nhưng toàn bộ < δ_min → vẫn đúng
    assert judge("<=0", 0.005, 0.02) == "inconclusive"         # lo = δ_min không > δ_min
    assert judge_iso("+", dict(gate_cheaper=True)) == "correct" and judge_iso("<=0", dict(gate_cheaper=True)) == "wrong"
    assert judge_iso("+", dict(gate_cheaper=True, diff_lo=-0.01, diff_hi=0.02)) == "inconclusive"
    assert judge_iso("+", dict(gate_cheaper=True, diff_lo=-0.03, diff_hi=-0.01)) == "correct"
    cells = [dict(id="a", confirmatory=True, comparison="matched_cost", sign_pred="+", hypothesis="H3/H5"),
             dict(id="b", confirmatory=True, comparison="matched_cost", sign_pred="<=0", hypothesis="H3/H5"),
             dict(id="c", confirmatory=True, comparison="matched_cost", sign_pred="<=0", hypothesis="H3/H5"),
             dict(id="h", confirmatory=True, comparison="r_pair", sign_pred="+", hypothesis="H8"),
             dict(id="x", confirmatory=True, comparison="r_pair", sign_pred="—", hypothesis="H8"),       # ngoài r† → không chấm
             dict(id="d", confirmatory=False, comparison="matched_cost", sign_pred="+", hypothesis="H3/H5")]
    obs = [dict(id="a", delta=0.2, lo=0.1, hi=0.3, p_boot=0.001), dict(id="b", delta=0.0, lo=-0.01, hi=0.01, p_boot=0.9),
           dict(id="c", delta=0.1, lo=0.05, hi=0.2, p_boot=0.001), dict(id="h", delta=0.02, lo=0.01, hi=0.03),
           dict(id="x", delta=0.0, lo=-1, hi=1), dict(id="d", delta=0.1, lo=0.05, hi=0.2, p_boot=0.001)]
    r = score(cells, obs)
    assert r["family_size"] == 4 and r["n_scored"] == 4 and r["n_decided"] == 3
    assert abs(r["share_correct"] - 2 / 3) < 1e-12 and r["criteria"]["i_reject"] and r["verdict"] == "rejected"
    assert r["three_way_by_group"]["H3-H5"]["inconclusive"] == 1 / 3 and r["three_way_by_group"]["H8"]["correct"] == 1.0
    assert r["n_sig_holm"] == 2                                          # a, c có ý nghĩa sau Holm (chỉ báo cáo)
    # tiêu chí công suất: > 50 % chưa kết luận → "chưa kiểm được" (kể cả khi phần kết luận được đều đúng/sai)
    cells2 = [dict(id=k, confirmatory=True, comparison="matched_cost", sign_pred="<=0", hypothesis="H3/H5") for k in "pqr"]
    o2 = [dict(id="p", delta=0, lo=-0.1, hi=0.1), dict(id="q", delta=0, lo=-0.1, hi=0.1), dict(id="r", delta=0.2, lo=0.1, hi=0.3)]
    r2 = score(cells2, o2)
    assert r2["criteria"]["P_untestable"] and r2["verdict"] == "untestable" and r2["criteria"]["i_reject"]
    o3 = [dict(id="p", delta=0, lo=-0.1, hi=0.001), dict(id="q", delta=0, lo=-0.1, hi=0.1), dict(id="r", delta=0, lo=-0.1, hi=0.0)]
    assert score(cells2, o3)["verdict"] == "confirmed"                 # 1/3 chưa kết luận, 2/2 kết luận được đúng


def test_holm_and_paired():
    adj, rej = holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5]) and list(rej) == [True, False, False, False]
    rng = np.random.default_rng(1)
    seq = np.repeat(np.arange(10), 30)
    mP, mG = rng.random(300) * 0.5 + 0.3, rng.random(300) * 0.3
    d = paired_delta(mP, mG, seq, 10, boot_weights(10, B=500))
    assert d["lo"] > 0 and d["sign"] == "+"


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
