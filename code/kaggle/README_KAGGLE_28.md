# README — Chạy notebook train YOLO26 trên Kaggle (Đề tài 28)

File: `28_train_yolo26_visdrone.ipynb` (một notebook, tự chứa, không cần file nào khác).

## Chọn model (quyết định 27-09-2026)
- **Detector chính: `yolo26s`** — chạy version 1.
- **Mức recall thấp: `yolo26n`** — chạy version 2 (đổi `MODEL = "yolo26n.pt"` ở Cell 2). Bài dùng 2 model × 3 độ phân giải export (640/960/1280) = 6 mức recall r cho trục "chất lượng detector".
- Không train `yolo26m`: chậm trên laptop Iris Xe, không cần cho câu hỏi nghiên cứu.

## Các bước
1. Vào kaggle.com → **Code** → **New Notebook** → **File → Import Notebook** → chọn `28_train_yolo26_visdrone.ipynb`.
2. Thanh phải **Settings**:
   - **Accelerator:** `GPU P100` nếu có, không thì `GPU T4 x2` (notebook chỉ dùng GPU 0).
   - **Internet:** `On`. Nếu bị xám: tài khoản chưa xác minh số điện thoại → Settings tài khoản → Phone verification.
   - **Persistence:** `Files only`.
3. Kiểm Cell 2: `MODEL = "yolo26s.pt"`, `SMOKE = False`.
4. **Save Version → Save & Run All (Commit)** → Save. Notebook chạy nền; đóng trình duyệt được. Xem tiến độ ở biểu tượng chuông / trang notebook → tab Logs.
5. Xong (ước 4–7 h): mở version → tab **Output** → tải `28_yolo26s_visdrone_1024_export.zip` (< 1 GB).
6. Đặt zip vào `$THS_DATASETS\_models\28\_zips\`, giải nén vào `$THS_DATASETS\_models\28\yolo26s_visdrone_1024\` (= `paths.model_dir("yolo26s_visdrone_1024")`; ngoài vùng Google Drive — luật Dataset 28-09-2026). Trước khi giải nén: `Get-PSDrive D` còn ≥ 1,5 × dung lượng + 20 GB; sau khi đối chiếu số entry thì xoá zip.
7. Lặp bước 3–6 với `MODEL = "yolo26n.pt"` (≈ 2–4 h) → `$THS_DATASETS\_models\28\yolo26n_visdrone_1024\`.
8. Báo Claude Code chạy `code\p2_verify_weights.py` (kiểm sha256 theo `TRAIN_LOG.md`, chạy thử OpenVINO trên 1 ảnh).

## Nếu phiên bị cắt ở 12 h
Mở notebook → **Add Input → Your Work** → chọn version vừa chạy dở → Save & Run All lại. Cell 4 tìm `last.pt` trong `/kaggle/input/**` và resume; các epoch đã xong không chạy lại.

## Version 1 chưa có nms=False → chạy lại val+export
Version 1 (`yolo26s`) được chạy với notebook cũ: `ultralytics` không ghim phiên bản và chưa có `nms=False` / `MAX_DET`. Với ultralytics 8.4.163, khi không truyền `nms=False` thì val dùng head one-to-many + NMS, và file export có output `(1, 84, N)` (vẫn cần NMS), không phải end-to-end. **Không cần train lại**, chỉ chạy lại val + export từ `best.pt` của version 1:
1. Mở notebook trên Kaggle → **File → Import Notebook** → chọn bản `28_train_yolo26_visdrone.ipynb` mới (đã ghim `ultralytics==8.4.163`, `openvino==2026.4.0`; có `SKIP_TRAIN`, `MAX_DET = 400`). Import sẽ ghi đè nội dung notebook; các version cũ vẫn còn trong Version history.
2. Thanh phải → **Add Input** → tab **Your Work** → chọn chính notebook này → chọn **Version 1** (version đã train xong) → **Add**. Output của version 1 sẽ nằm dưới `/kaggle/input/...`.
3. Cell 2: giữ `MODEL = "yolo26s.pt"`, `IMGSZ = 1024`, `SMOKE = False` (để `NAME` = `yolo26s_visdrone_1024` khớp version 1), rồi đặt **`SKIP_TRAIN = True`**. Giữ `MAX_DET = 400`.
4. Settings: GPU bật, Internet `On` (vẫn phải tải lại VisDrone để val). **Save Version → Save & Run All (Commit)**. Mất khoảng **15 phút** (tải dữ liệu, val, 6 lần export).
5. Cell 4 tự tìm `best.pt` của `yolo26s_visdrone_1024` trong `/kaggle/input/**`: thư mục `runs/<NAME>/weights/`, hoặc bản zip đã giải nén, hoặc `28_<NAME>_export.zip` nguyên. Nếu không thấy, cell 4 báo lỗi rõ ràng (kiểm lại bước 2–3).
6. Xong → tab **Output** của version mới → tải `28_yolo26s_visdrone_1024_export.zip` **mới** (thay cho zip của version 1), giải nén vào `$THS_DATASETS\_models\28\yolo26s_visdrone_1024\` (xóa bản cũ trước), rồi chạy `code\p2_verify_weights.py`. Script này giờ kiểm thêm "no-NMS export": mọi IR/ONNX phải có output `[1, 400, 6]` và không có op NMS. Zip cũ của version 1 sẽ FAIL ở bước này.
7. Trong version này, `TRAIN_LOG.md` ghi "Thời gian train: nan h" vì không train. Thời gian train thật xem `TRAIN_LOG.md` của version 1.

**yolo26n:** chạy như bước 3–7 ở mục "Các bước" với notebook mới đã ghim phiên bản (`MODEL = "yolo26n.pt"`, `SKIP_TRAIN = False`, `MAX_DET = 400`). Không cần bước chạy lại ở trên.

## Chạy thử nhanh (tùy chọn, 10–15 phút)
Đặt `SMOKE = True` → Save & Run All. Nó train 1 epoch trên 2% dữ liệu ở 320 px để bắt lỗi môi trường; kết quả không dùng cho bài. Nhớ đặt lại `SMOKE = False`.

## Trong zip có gì
`weights/best.pt`, `weights/last.pt` · `export/*_openvino_{640,960,1280}/` (FP16) + `*.onnx` · `val_metrics.json` (mAP trên VisDrone-DET val) · `results.csv`, `args.yaml`, hình train/val · `dataset_manifest.json` (nguồn tải, số ảnh, sha256 zip dữ liệu) · `TRAIN_LOG.md` (GPU, phiên bản, thời gian, sha256 mọi file).

## Ghi chú cho bài báo
- Dữ liệu train là **VisDrone2019-DET** (ảnh tĩnh), khác tập đánh giá **VisDrone-VID / UAVDT**. Cần kiểm trùng cảnh DET-train ↔ VID-test trước khi báo số (việc của P2).
- YOLO26 end-to-end **không có bước NMS** → chi phí mỗi lần chạy detector gần như hằng số theo độ phân giải, thuận cho mô hình chi phí (activation × cost) của bài; ghi rõ điều này trong phần thiết lập.
- Ghi vào bài: phiên bản `ultralytics`, seed 42, `deterministic=True`, GPU đã dùng, số epoch thực đạt (trong `TRAIN_LOG.md`).
