# DATA_SOURCES — P0 #28 (sinh bởi code/p0_report.py)

Chỉ tải từ link Google Drive chính thức (VisDrone: README github.com/VisDrone/VisDrone-Dataset; UAVDT: sites.google.com/view/grli-uavdt). Không re-host frame; dữ liệu nằm ngoài repo (THS_DATASETS — xem code/paths.py). Link bị chặn: xem DOWNLOAD_BLOCKED.md.

Vị trí (từ 28-09-2026): UAVDT annotation → `$THS_DATASETS/UAVDT/annotations/`, frames → `$THS_DATASETS/UAVDT/frames/UAV-benchmark-M/`, zip → `$THS_DATASETS/UAVDT/_zips/`, `$THS_DATASETS/VisDrone2019-VID/_zips/`; VisDrone-VID → `$THS_DATASETS/VisDrone2019-VID/{train,val,test-dev}/` (gốc THS_DATASETS). sha256 dưới đây là của zip gốc (Move-Item cùng ổ không đổi byte); zip có thể đã xoá sau đối chiếu — bản ghi giữ trong results/p0_stats.json.

| File | Nội dung | URL | Ngày tải | Kích thước (byte) | sha256 | License/điều khoản |
|---|---|---|---|---|---|---|
| UAVDT_MOT_toolkit.zip | UAVDT UAV-benchmark-MOTD_v1.0 (GT + toolkit) | https://drive.google.com/open?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq | 2026-09-25 | 245719325 | `1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799` | GNU-GPL (toolkit); dataset "research purpose only" (README) |
| UAVDT_attributes.zip | UAVDT sequence attributes (M_attr) | https://drive.google.com/open?id=1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1 | 2026-09-25 | 9503 | `ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1` | như UAVDT |
| UAV-benchmark-M.zip | UAVDT frames UAV-benchmark-M | https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view | — | — | **CHƯA TẢI (quota Drive)** | research only |
| VisDrone2019-VID-train.zip | VisDrone2019-VID train | https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw/view | — | — | **CHƯA TẢI (quota Drive)** | research only (điều khoản AISKYEYE) |
| VisDrone2019-VID-val.zip | VisDrone2019-VID val | https://drive.google.com/file/d/1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B/view | — | — | **CHƯA TẢI (quota Drive)** | research only |
| VisDrone2019-VID-test-dev.zip | VisDrone2019-VID test-dev | https://drive.google.com/open?id=1-BEq--FcjshTF1UwUabby_LHhYj41os5 | — | — | **CHƯA TẢI (quota Drive)** | research only |

## Số thực đếm vs kỳ vọng

| Dataset | Mục | Thực đếm | Kỳ vọng | Ghi chú |
|---|---|---|---|---|
| VisDrone-VID | clip / khung | CHƯA CÓ DỮ LIỆU | 79 / 33366 | quota Drive — DOWNLOAD_BLOCKED.md |
| UAVDT | chuỗi (train/test) | 50 (30/20) | ~50 | |
| UAVDT | khung | 40421 | ~40000 | đếm ảnh |
| UAVDT | kích thước ảnh | 1024x540(image), 960x540(image) | 1080×540 | lệch — xem dưới |
| UAVDT | track xe / box | 2653 / 798795 | — | |

### Độ lệch UAVDT so với kỳ vọng / README (L5 — ghi lại, không ép)
- Số chuỗi DET/MOT thực: 50 (train 30 / test 20); KE_HOACH mục 4 ghi 100 seq ~80k khung — đó là tổng gồm cả phần SOT (UAV-benchmark-S), không thuộc P0.
- Số khung: 40421 (nguồn: images; = frame index lớn nhất trong GT mỗi chuỗi khi chưa có ảnh).
- Kích thước ảnh: kỳ vọng 1080×540 (paper). GT extent lớn nhất = 1025×541 → ảnh thực nhiều khả năng 1024×540. Đang dùng 1024x540(image), 960x540(image); xác nhận khi có frames.
- README MOTD: file *_gt.txt ghi `<in-view>` và `<occlusion>` "hằng -1", thực tế cột 8 = 1, cột 9 = -1. P0 dùng *_gt_whole.txt (đúng định dạng README: out_of_view ∈ {1,2,3}, occlusion ∈ {1,2,3,4}, category ∈ {1,2,3}).
- File thuộc tính test có tên lỗi `M0701 _attr.txt` (dấu cách) — parser đã strip.
- Box theo loại: {'car': 755688, 'truck': 25086, 'bus': 18021}; occlusion: {'1': 715470, '4': 64004, '3': 10025, '2': 9296}; out_of_view: {'1': 725059, '2': 37089, '3': 36647}.
- Độ cao (thuộc tính chuỗi): {'medium': 29, 'low': 18, 'high': 3}.
