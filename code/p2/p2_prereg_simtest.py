"""P2B-B4 / P2C-B3: thử p2_prereg_build + p2_prereg_score với KÊNH GIẢ LẬP lấy từ A7 — không phải PREREG thật.

Kênh giả (P2C): 4 cue giả = 4 đường ROC đi qua lưới (q_in, q_out) của A7 ở w = 5 (PSEUDO_CUES; θ = chỉ số điểm, θ lớn = ngưỡng thấp);
builder chọn θ* theo luật B1 (Youden, a_G(R=30) ≤ 0,25) rồi dựng họ thu gọn (≤ 80 ô). ρ_onset = giá trị TRAIN trong A7; c = 0;
r_levels = {sim_r1: 1, sim_r0.8: 0.8, sim_r0.5: 0.5}, detector chính sim_r1, cặp H8 (r1, r0.8), (r1, r0.5);
q_b chỉ là nhãn "unknown" (q_o = None → q_out dùng trực tiếp; không có trục ego).
Quan sát:
  matched cost  ← results/p2/sim/agnostic_cells.csv (fp = 0, E3, theo M_bin): delta, lo, hi, p_boot
  iso-KPI       ← gate đạt KPI (miss_G ≤ ε, từ agnostic_cells, M_bin = all) và a_G < a_P(ε) (engine periodic_iso, TRAIN) — ε ∈ {1,2,5 %}
                  (quan sát iso ở mức "mọi M"; áp cho cả 3 ô M_bin của builder — xấp xỉ, ghi rõ)
  H8            ← chênh Δ giữa hai mức r cùng cấu hình (M_bin = all, áp cho các ô M_bin), CI bảo thủ [lo_lo − hi_hi, hi_lo − lo_hi]
  ranh giới     ← results/p2/sim/mstar_iso.json: q_o* replay vs CI dự đoán, chỉ các ô có thông tin (0 < q_o* < ∞)
Đầu ra: results/p2/sim/prereg_sim_cells.json, PREREG_SIM_TEST.md, prereg_sim_score.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "replay"))
from p2_prereg_build import build, cell_id, train_events, write_md  # noqa: E402
from p2_prereg_score import score  # noqa: E402

SIM = ROOT / "results" / "p2" / "sim"
RL = {1.0: "sim_r1", 0.8: "sim_r0.8", 0.5: "sim_r0.5"}
# cue giả: đường ROC (q_in, q_out) qua lưới A7, θ = chỉ số điểm (0 = ngưỡng cao nhất). Cố định trước khi chạy.
PSEUDO_CUES = {
    "simA_clean": [(0.2, 0.0), (0.5, 0.0), (0.8, 0.01), (1.0, 0.02)],
    "simB_mid": [(0.2, 0.0), (0.5, 0.01), (0.8, 0.02), (1.0, 0.05)],
    "simC_noisy": [(0.2, 0.01), (0.5, 0.02), (0.8, 0.05), (1.0, 0.10)],
    "simD_poor": [(0.2, 0.02), (0.5, 0.05), (0.8, 0.10)],
}
W_SIM = 5


def iso_aP(eps_list, kpis=(3, 5, 10, 30)):
    """a_P(ε) từ engine (periodic S thực, 20 pha) trên TRAIN, detector giả lập fp=0."""
    from p2_dump_schema import DUMPS
    from p2_engine import EventSet, Timeline, load_events, periodic_iso
    from p2_match import hits_for_dump
    from p2_sim_detector import dump_name
    P0 = ROOT / "results" / "p0"
    fr = pd.read_parquet(P0 / "frames_uavdt.parquet")
    tl = Timeline(fr[fr.split == "train"].groupby("seq").frame.max().to_dict())
    rows = pd.read_parquet(P0 / "rows_uavdt.parquet", columns=["split", "seq", "frame", "tid", "x", "y", "w", "h", "group"])
    rows = rows[(rows.group == "VEHICLE") & (rows.split == "train")]
    ev = load_events("train")
    cache = SIM / "iso_aP_engine.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    for r in RL:
        es = None
        for k in kpis:
            for e in eps_list:
                key = f"{r}|{k}|{e}"
                if key in have:
                    continue
                if es is None:
                    es = EventSet(ev["E3"], hits_for_dump(DUMPS / dump_name(r, 0.0), rows), tl)
                S, a, _ = periodic_iso(tl, es, e, "lat", k)
                have[key] = a if a is not None else math.inf
    cache.write_text(json.dumps(have, indent=1, default=float))
    return have


def main():
    info = json.loads((SIM / "theory_repro.json").read_text(encoding="utf-8"))["info"]
    cdf = pd.read_csv(SIM / "agnostic_cells.csv")
    ops = [dict(cue=cue, theta=k, w=W_SIM, q_in=float(qi), q_out=float(qo), q_b={"unknown": 0.0}, q_o=None, c=0.0)
           for cue, roc in PSEUDO_CUES.items() for k, (qi, qo) in enumerate(roc)]
    lam = float(pd.read_parquet(ROOT / "results" / "p0" / "events_E3_uavdt.parquet")
                .query("split=='train' and vis_def=='main' and not never_visible").shape[0] / info["n_frames"])
    channel = dict(split="train", simulated=True, dataset="UAVDT", lambda_per_frame=lam,
                   rho_onset={str(k): v for k, v in info["rho_onset"].items()}, operating_points=ops, r_levels={v: k for k, v in RL.items()},
                   main_level="sim_r1", h8_pairs=[["sim_r1", "sim_r0.8"], ["sim_r1", "sim_r0.5"]])
    binfo = {}
    cells = build(channel, train_events(), info=binfo)
    (SIM / "prereg_sim_cells.json").write_text(json.dumps(dict(simulated=True, n=len(cells), info=binfo, cells=cells), indent=1, default=float),
                                               encoding="utf-8")
    write_md(cells, channel, SIM / "PREREG_SIM_TEST.md", binfo)
    base = cdf[(cdf.fp == 0.0) & (cdf.event == "E3") & (cdf.w == W_SIM)]
    Rv = lambda x: math.inf if str(x) == "inf" else int(float(x))  # noqa: E731
    roc_all = {}
    for cue, pts in PSEUDO_CUES.items():
        for k, (qi, qo) in enumerate(pts):
            roc_all.setdefault((qi, qo), []).append((cue, k))
    obs = []
    for x in base[base.M_bin != "all"].itertuples(index=False):
        kpi = x.kpi if x.kpi == "miss" else int(x.kpi)
        for cue, k in roc_all.get((x.q_in, x.q_out), []):
            obs.append(dict(id=cell_id("UAVDT", 0.0, kpi, x.M_bin, "unknown", RL[x.r], cue, W_SIM, Rv(x.R), k),
                            delta=x.delta, lo=x.lo, hi=x.hi, p_boot=x.p_boot))
    aP = iso_aP([0.01, 0.02, 0.05])
    allM = base[base.M_bin == "all"]
    for x in allM[allM.kpi.astype(str).isin(["3", "5", "10", "30"])].itertuples(index=False):
        for e in (0.01, 0.02, 0.05):
            cheaper = bool(x.miss_G <= e + 1e-12 and x.a_G < aP[f"{x.r}|{int(x.kpi)}|{e}"])
            for cue, k in roc_all.get((x.q_in, x.q_out), []):
                for mb in ("1-5", "6-20", ">20"):
                    obs.append(dict(id=cell_id("UAVDT", 0.0, f"iso_L{int(x.kpi)}_eps{e:g}", mb, "unknown", RL[x.r], cue, W_SIM, Rv(x.R), k),
                                    gate_cheaper=cheaper))
    for (qi, qo, R, kpi), g in allM[allM.kpi.astype(str).isin(["3", "5", "10"])].groupby(["q_in", "q_out", "R", "kpi"]):
        g = g.set_index("r")
        for hi, lo in ((1.0, 0.8), (1.0, 0.5), (0.8, 0.5)):
            d = g.loc[lo, "delta"] - g.loc[hi, "delta"]
            ci = (g.loc[lo, "lo"] - g.loc[hi, "hi"], g.loc[lo, "hi"] - g.loc[hi, "lo"])
            for cue, k in roc_all.get((qi, qo), []):
                for mb in ("1-5", "6-20", ">20", "all"):   # "all" = ô H8 gộp M (bước thu gọn H8_pool_M)
                    obs.append(dict(id=cell_id("UAVDT", 0.0, f"H8_L{int(kpi)}", mb, "unknown", f"{RL[hi]}>{RL[lo]}", cue, W_SIM, Rv(R), k),
                                    delta=float(d), lo=float(ci[0]), hi=float(ci[1])))
    mi = json.loads((SIM / "mstar_iso.json").read_text(encoding="utf-8"))["rows"]
    boundary = [dict(name=f"r{x['r']}|{x['group']}|L{x['Lmax']}|eps{x['eps']}", obs=x["qo_star_replay"], pred=x["qo_star_pred"],
                     pred_ci=x["qo_star_pred_ci"])
                for x in mi if 0 < x["qo_star_pred"] < math.inf and x["qo_star_pred_ci"] is not None and np.isfinite(x["qo_star_replay"])]
    res = score(cells, obs, boundary)
    wrong = [r for r in res["rows"] if not r["correct"]]
    res["wrong_by_hypothesis_obs"] = pd.DataFrame(wrong).groupby(["hypothesis", "pred", "obs"]).size().reset_index(name="n").to_dict("records") if wrong else []
    res["theta_star"] = binfo["theta_star"]
    res["family"] = binfo["family"]
    (SIM / "prereg_sim_score.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print({k: v for k, v in res.items() if k not in ("rows", "theta_star")})
    return res


if __name__ == "__main__":
    main()
