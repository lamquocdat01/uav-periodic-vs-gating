# ENV_28 — Môi trường P0 (ghi 2026-09-25)

| Mục | Giá trị |
|---|---|
| OS | Microsoft Windows 11 Pro 10.0.26200 (build 26200) |
| CPU | 11th Gen Intel Core i7-1185G7 @ 3.00 GHz — 4 nhân / 8 luồng |
| RAM | 31.7 GB |
| GPU | Intel Iris Xe Graphics (tích hợp, ~2 GB chia sẻ). **Không có GPU NVIDIA** — `nvidia-smi` không tồn tại, không có driver/CUDA |
| Ổ D: trống | 76.2 GB (≥ 40 GB → đủ cho bước tải) |
| Python | 3.11.9, venv `.venv\` |
| Gói | numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, scipy 1.17.1, opencv-python 5.0.0, gdown 6.4.0, tqdm, requests |
| Git | 2.52.0.windows.1 (chỉ local) |

## ⚠ Lưu ý cho P2
KE_HOACH mục 1/4 giả định "GPU laptop" (train YOLO 6–10 h/model). Máy này **không có GPU rời CUDA** → train/dump YOLO ở P2 trên CPU sẽ chậm hơn nhiều bậc. Cần anh Đạt chốt ở CP0: máy GPU khác (không phải Orin) hay dùng trọng số pretrained + fine-tune nhỏ.

## Cập nhật 2026-09-27 (P0b-C / P2A)
| Mục | Giá trị |
|---|---|
| Gói mới trong .venv | ultralytics 8.4.163, openvino 2026.4.0, torch 2.14.0+cpu, onnx 1.23.0, onnxslim 0.1.96, onnxruntime 1.30.0, nbconvert 7.17.1, nbclient 0.11.0, ipykernel 7.3.0, matplotlib 3.11.2 (setuptools 65.5 → 84.0). numpy/pandas/scipy/opencv/pyarrow giữ nguyên (cài với constraints). Chi tiết: results/p2/env_p2.json |
| OpenVINO devices | CPU + GPU (Iris Xe) — benchmark: results/p2/BENCH_OPENVINO.md |
| TeX | MiKTeX 25.12 (pdfTeX 4.23) có sẵn tại `<home>\AppData\Local\Programs\MiKTeX\miktex\bin\x64\` (không trên PATH Git-Bash); không cài/cập nhật gì. MiKTeX báo "chưa kiểm cập nhật" — anh Đạt quyết |
| Python hệ thống | có ultralytics 8.4.7 riêng (không dùng) — kernel Jupyter phải chạy bằng .venv |
| Máy P2 (đã chốt 25/09) | Train YOLO26 trên Kaggle; dump + cue trên laptop OpenVINO (iGPU nhanh hơn CPU mọi tổ hợp). Mục "⚠ Lưu ý cho P2" ở trên đã được giải quyết bằng quyết định này |
| TeX (P2B, 27/09) | MiKTeX-pdfTeX 4.23 (MiKTeX 25.12) — **không cập nhật** theo quyết định anh Đạt 27/09 |
| Ghim notebook Kaggle | ultralytics==8.4.163, openvino==2026.4.0 (cell 1); export max_det=400, nms=False ⇒ output [1, 400, 6] |
| Weights yolo26s chính thức (P2D, 29/09) | `$THS_DATASETS\_models\28\yolo26s_visdrone_1024\` = Kaggle **Version 2** (`code/kaggle/28_v2_yolo26s_reexport.ipynb`, SKIP_TRAIN, 2026-09-29, Tesla T4, ultralytics 8.4.163): best.pt sha256 825dffcc…bf84 (giống v1), val mAP50 0,5001 / mAP50-95 0,3057 (max_det=400); export/ IR+ONNX 640/960/1024/1280 no-NMS [1,400,6]. Version 1 (export có NMS) giữ ở `yolo26s_visdrone_1024_v1_nms\`; bản xuất cục bộ P2C ở `export_local_reference\` (chỉ tham khảo) |
| Weights yolo26n (P2E, 30/09) | `$THS_DATASETS\_models\28\yolo26n_visdrone_1024\` = Kaggle **Version 3** (`code/kaggle/28_v3_yolo26n_train.ipynb`, train thật 80/80 epoch, 6,02 h, 2026-09-29 13:06 UTC, Tesla T4, ultralytics 8.4.163, cùng VisDrone2019-DET như yolo26s): best.pt sha256 4dc79ab7…72e3; val mAP50 0,4041 / mAP50-95 0,2399 (max_det=400); export/ IR+ONNX 640/960/1024/1280 no-NMS [1,400,6]. `export_local\yolo26n_visdrone_1024_openvino_320\` = IR cue tiny xuất cục bộ (P2E, .venv ultralytics 8.4.163 / openvino 2026.4.0, half, nms=False, max_det=400). yolo26n COCO chỉ còn ở `p2_test\` (unit test) và `p2_bench\` (bench không --real) |
