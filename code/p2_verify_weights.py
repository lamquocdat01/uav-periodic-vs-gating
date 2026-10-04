"""Verify the Kaggle export (28_y8s_visdrone_export.zip unpacked) before use in P2.

Works with the export of code/kaggle/28_train_yolo26_visdrone.ipynb (zip unpacked into paths.MODELS_DIR/<NAME>/, i.e. THS_DATASETS/_models/28/<NAME>/):
  weights/best.pt, weights/last.pt, export/<NAME>_openvino_<sz>/*.xml|bin, export/<NAME>_<sz>.onnx, TRAIN_LOG.md ...

1. sha256 of every file listed in TRAIN_LOG.md must match the file on disk. Accepted line formats:
   "<sha256>  <relpath>" (sha256sum style, as written by the notebook) or a markdown row
   "| relpath | bytes | sha256 |". Optionally also sha256_manifest.json {relpath: {sha256, bytes}}.
2. Every OpenVINO IR (*.xml, one per imgsz) is compiled on CPU and run on 1 synthetic 1024x540 image
   (+ ultralytics' bundled bus.jpg if available); box count (conf >= 0.25) and inference time are printed.
   YOLO26 IR output is end-to-end (1, 300, 6) = [x1, y1, x2, y2, score, cls] -> no NMS; a classic
   (1, 4+nc, N) head is also handled (with NMS). --onnx additionally runs the .onnx files via OpenVINO.
3. "No-NMS export" check (static, every IR *.xml and every *.onnx found): FAIL if any op type contains
   "NMS"/"NonMaxSuppression" (case-insensitive), or the model does not have exactly one output of shape
   [1, MAX_DET, 6]. MAX_DET defaults to 400 (--max-det to override). This catches both a default
   nms=None export (output (1, 4+nc, N)) and an nms=True export (same [1, k, 6] shape but with an NMS op).

Usage:
  .venv/Scripts/python.exe code/p2_verify_weights.py --name yolo26s_visdrone_1024
  .venv/Scripts/python.exe code/p2_verify_weights.py --dir <folder> [--zip export.zip] [--onnx] [--json-out f.json]
  .venv/Scripts/python.exe code/p2_verify_weights.py --dir <folder> --skip-hash --max-det 400   (no-NMS check only)
Exit code: 0 = all OK; 2 = hash mismatch/missing; 3 = no-NMS export check failed; 1 = other failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from paths import model_dir  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_train_log(p: Path) -> dict:
    tab = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\|\s*`?([^|`]+?)`?\s*\|\s*(\d+)\s*\|\s*`?([0-9a-fA-F]{64})`?\s*\|", line)
        if m:
            tab[m.group(1).strip().replace("\\", "/")] = {"bytes": int(m.group(2)), "sha256": m.group(3).lower()}
            continue
        m = re.match(r"^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if m:
            tab[m.group(2).strip().replace("\\", "/")] = {"bytes": None, "sha256": m.group(1).lower()}
    return tab


def letterbox(img, s_h, s_w):
    import cv2
    h, w = img.shape[:2]
    r = min(s_h / h, s_w / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(img, (nw, nh))
    top, left = (s_h - nh) // 2, (s_w - nw) // 2
    return cv2.copyMakeBorder(im, top, s_h - nh - top, left, s_w - nw - left, cv2.BORDER_CONSTANT,
                              value=(114, 114, 114))


def count_boxes(out: np.ndarray, conf=0.25, iou=0.7) -> int:
    import cv2
    p = out[0]
    if p.ndim == 2 and p.shape[1] == 6:  # YOLO26 end-to-end: (300, 6) [x1,y1,x2,y2,score,cls], NMS-free
        return int((p[:, 4] >= conf).sum())
    # classic head (4+nc, N)
    if p.shape[0] > p.shape[1]:
        p = p.T
    boxes, scores = p[:4].T, p[4:].max(0)
    keep = scores >= conf
    if not keep.any():
        return 0
    b = boxes[keep]
    xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores[keep].tolist(), conf, iou)
    return int(len(idx))


def synth_image(seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(0, 1, (540, 1024, 3)).astype(np.float32), (0, 0), 3)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, 1000)), int(rng.integers(0, 520))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(6, 20))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


NMS_KEYS = ("nms", "nonmaxsuppression")


def check_no_nms(path: Path, max_det: int, core=None) -> dict:
    """Static check that an OpenVINO IR (.xml) or ONNX export is the NMS-free end-to-end head [1, max_det, 6]."""
    rec = {"file": str(path), "expected_output": [1, max_det, 6]}
    if path.suffix.lower() == ".xml":
        net = core.read_model(str(path))
        ops = sorted({op.get_type_name() for op in net.get_ops()})
        outs = []
        for o in net.outputs:
            ps = o.get_partial_shape()
            outs.append([int(dm.get_length()) if dm.is_static else -1 for dm in ps])
        rec["format"] = "openvino"
    else:
        import onnx
        g = onnx.load(str(path), load_external_data=False).graph
        ops = set()

        def walk(graph):
            for n in graph.node:
                ops.add(n.op_type)
                for at in n.attribute:  # sub-graphs (If/Loop)
                    if at.g is not None and len(at.g.node):
                        walk(at.g)
                    for sg in at.graphs:
                        walk(sg)
        walk(g)
        ops = sorted(ops)
        outs = [[dm.dim_value if dm.HasField("dim_value") else -1 for dm in o.type.tensor_type.shape.dim] for o in g.output]
        rec["format"] = "onnx"
    nms_ops = [t for t in ops if any(k in t.lower() for k in NMS_KEYS)]
    rec.update({"outputs": outs, "nms_ops": nms_ops, "has_topk": any(t.lower() == "topk" for t in ops)})
    problems = []
    if nms_ops:
        problems.append(f"NMS op(s) present: {nms_ops}")
    if len(outs) != 1:
        problems.append(f"{len(outs)} outputs (expected 1)")
    elif outs[0] != [1, max_det, 6]:
        problems.append(f"output shape {outs[0]} != [1, {max_det}, 6]")
    rec["problems"] = problems
    rec["pass"] = not problems
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo26s_visdrone_1024", help="folder name under ROOT/models")
    ap.add_argument("--dir", default=None, help="explicit folder (overrides --name)")
    ap.add_argument("--onnx", action="store_true", help="also run the .onnx files")
    ap.add_argument("--zip", default=None, help="optional: unzip this export zip into --dir first")
    ap.add_argument("--json-out", default=None, help="optional: write verification result json here")
    ap.add_argument("--max-det", type=int, default=400, help="expected max_det of the end-to-end export (default 400)")
    ap.add_argument("--skip-hash", action="store_true", help="skip the sha256 check (e.g. to test a bare export)")
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else model_dir(a.name)
    if a.zip:
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(d)
        print(f"[unzip] {a.zip} -> {d}")
    if not d.exists():
        sys.exit(f"[error] directory not found: {d}")

    ok = True
    report = {"dir": str(d), "hash": {}, "models": []}
    # ---------------- hashes
    tables = {}
    if (d / "sha256_manifest.json").exists():
        tables["sha256_manifest.json"] = json.loads((d / "sha256_manifest.json").read_text(encoding="utf-8"))
    if (d / "TRAIN_LOG.md").exists():
        tables["TRAIN_LOG.md"] = parse_train_log(d / "TRAIN_LOG.md")
    if a.skip_hash:
        tables = {}
        report["hash_ok"] = True
        print("[hash] skipped (--skip-hash)")
    elif not tables:
        print("[error] neither sha256_manifest.json nor TRAIN_LOG.md found")
        report["hash_ok"] = False
        ok = False
    for src, tab in tables.items():
        n_ok = n_bad = n_miss = 0
        for rel, ref in tab.items():
            f = d / rel
            if not f.exists():
                n_miss += 1
                print(f"[MISSING] ({src}) {rel}")
                continue
            if sha256(f) == ref["sha256"] and (ref["bytes"] is None or f.stat().st_size == ref["bytes"]):
                n_ok += 1
            else:
                n_bad += 1
                print(f"[MISMATCH] ({src}) {rel}")
        print(f"[hash] {src}: {n_ok} OK, {n_bad} mismatch, {n_miss} missing (of {len(tab)})")
        report["hash"][src] = {"ok": n_ok, "mismatch": n_bad, "missing": n_miss, "listed": len(tab)}
        hash_ok = (n_bad == 0 and n_miss == 0 and len(tab) > 0)
        ok &= hash_ok
        report["hash_ok"] = report.get("hash_ok", True) and hash_ok
    if len(tables) == 2:
        same = tables["sha256_manifest.json"] == tables["TRAIN_LOG.md"]
        print(f"[hash] manifest json and TRAIN_LOG.md tables identical: {same}")
        report["hash"]["tables_identical"] = same

    # ---------------- no-NMS export check (static)
    import cv2
    import openvino as ov
    core = ov.Core()
    report["no_nms"] = []
    static = sorted(p for p in d.rglob("*.xml") if "openvino" in p.parent.name.lower()) + sorted(d.rglob("*.onnx"))
    for f in static:
        try:
            r = check_no_nms(f, a.max_det, core)
        except Exception as e:
            r = {"file": str(f), "pass": False, "problems": [f"cannot read: {e!r}"]}
        r["file"] = str(f.relative_to(d))
        report["no_nms"].append(r)
        print(f"[no-NMS] {'PASS' if r['pass'] else 'FAIL'} {r['file']}: outputs {r.get('outputs')}, "
              f"NMS ops {r.get('nms_ops')}, TopK {r.get('has_topk')}" + (f" -> {'; '.join(r['problems'])}" if r["problems"] else ""))
    report["no_nms_ok"] = bool(static) and all(r["pass"] for r in report["no_nms"])
    if not static:
        print("[no-NMS] FAIL: no IR / onnx found")
    ok &= report["no_nms_ok"]

    # ---------------- run models
    imgs = {"synthetic_1024x540": synth_image()}
    try:
        from ultralytics.utils import ASSETS
        if (ASSETS / "bus.jpg").exists():
            imgs["ultralytics_bus.jpg"] = cv2.imread(str(ASSETS / "bus.jpg"))
    except Exception:
        pass
    models = sorted(p for p in d.rglob("*.xml") if "openvino" in p.parent.name.lower())
    if a.onnx:
        models += sorted(d.rglob("*.onnx"))
    if not models:
        print("[error] no OpenVINO IR / onnx found")
        ok = False
    for m in models:
        rec = {"model": str(m.relative_to(d))}
        try:
            net = core.read_model(str(m))
            shp = net.inputs[0].get_partial_shape()
            H, W = int(shp[2].get_length()), int(shp[3].get_length())
            cm = core.compile_model(net, "CPU")
            rec["input_hw"] = [H, W]
            for name, img in imgs.items():
                x = letterbox(img, H, W)[:, :, ::-1].transpose(2, 0, 1)
                x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
                cm({0: x})  # warm-up
                t0 = time.perf_counter()
                out = cm({0: x})[cm.outputs[0]]
                rec[f"ms_{name}"] = round((time.perf_counter() - t0) * 1e3, 2)
                rec[f"boxes_{name}"] = count_boxes(out)
                rec["output_shape"] = list(out.shape)
            print(f"[run] {rec['model']}: input {H}x{W}, out {rec['output_shape']}, "
                  + ", ".join(f"{k}={v}" for k, v in rec.items() if k.startswith(("boxes_", "ms_"))))
        except Exception as e:
            rec["error"] = repr(e)
            ok = False
            print(f"[run] {rec['model']}: ERROR {e!r}")
        report["models"].append(rec)
    report["all_ok"] = bool(ok)
    print("[RESULT]", "PASS" if ok else "FAIL")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if ok else (2 if not report.get("hash_ok", False) else (3 if not report["no_nms_ok"] else 1)))


if __name__ == "__main__":
    main()
