# DOSSIER_TITS_28 — Bài cùng dạng trên IEEE T-ITS (và TVT cho Plan B)

- **Cách tìm:** Crossref, lọc `issn:1524-9050 | issn:1558-0016` (T-ITS) và `issn:0018-9545 | issn:1939-9359` (TVT), `from-pub-date:2023-01-01`, `type:journal-article`.
- **Từ khoá truy vấn:** 11 truy vấn ghép (UAV/drone/aerial) × (traffic, vehicle detection, video, edge, onboard, efficient, frame). Kết quả thô nằm ở `results/p0/crossref_dossier_raw.json`: 210 DOI T-ITS và 188 DOI TVT.
- **Xác minh:** mọi DOI dưới đây đều được kiểm lại trên Crossref (khớp title/năm/venue) và resolve qua doi.org, lưu ở `results/p0/crossref_dossier_verified.json`. Abstract lấy từ OpenAlex, lưu ở `results/p0/dossier_abstracts.json`.
- **Tiêu chí "cùng dạng":** nghiên cứu thực nghiệm hoặc về hiệu quả, xoay quanh phát hiện hoặc giám sát xe từ video/ảnh UAV. Ưu tiên bài có yếu tố chi phí tính toán, độ cao hoặc tần số khung.

## T-ITS: đạt 9 bài cùng dạng + YOLC (yêu cầu ≥ 3)

| # | Title | Năm | DOI | Vì sao cùng dạng |
|---|---|---|---|---|
| T1 | YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images | 2024 | 10.1109/TITS.2024.3386928 | Phát hiện vật thể nhỏ trong ảnh aerial với tài nguyên tính toán hạn chế. Là bài đã có sẵn từ trước |
| T2 | A Review of Vision-Based Vehicle Detection for UAV-Based Traffic Monitoring: Experimental Insights and Future Directions | 2026 | 10.1109/TITS.2026.3696046 | **Sát nhất.** Đánh giá thực nghiệm độ chính xác và **độ trễ** của detector xe từ UAV; nêu thách thức về độ cao và bù chuyển động |
| T3 | Large-Scale High-Altitude UAV-Based Vehicle Detection via Pyramid Dual Pooling Attention Path Aggregation Network | 2024 | 10.1109/TITS.2024.3396915 | Phát hiện xe theo **độ cao bay** 250–400 m, gắn trực tiếp với P2 (độ cao tối ưu) |
| T4 | Lightweight Semantic Feature Extraction Model With Direction Awareness for Aerial Traffic Object Detection | 2026 | 10.1109/TITS.2025.3642410 | Detector giao thông aerial, trọng tâm là backbone nhẹ (chi phí tính toán) |
| T5 | Efficient Vehicle Recognition and Tracking for UAV-Enabled ITS: A Multi-Agent RL Method | 2025 | 10.1109/TITS.2025.3601740 | Nhận dạng và bám xe từ UAV với mục tiêu hiệu quả |
| T6 | Developing a More Reliable Framework for Extracting Traffic Data From a UAV Video | 2023 | 10.1109/TITS.2023.3290827 | Trích dữ liệu giao thông từ video UAV bằng YOLOv5-OBB + DeepSORT, có thực nghiệm hiện trường |
| T7 | Biases in Aerial Video-Based Vehicle Trajectory Generation: An Empirical Evaluation | 2026 | 10.1109/TITS.2026.3658135 | Đánh giá thực nghiệm sai lệch của pipeline video aerial. Cùng tinh thần "đo đạc/stress-test" |
| T8 | Hierarchical Spatial–Temporal UAV Tracking With 3-D Wavelets for Road Traffic Surveillance | 2025 | 10.1109/TITS.2025.3588075 | Tracking UAV cho giám sát giao thông, nhấn mạnh chi phí thời gian của mô-đun thời gian |
| T9 | A Multi-Task Framework for Car Detection From High-Resolution UAV Imagery Focusing on Road Regions | 2024 | 10.1109/TITS.2024.3432761 | Phát hiện xe từ ảnh UAV phân giải cao, giới hạn vùng xử lý vào mặt đường (ý "gating theo vùng") |
| T10 | SICNet: Structured Integrated Cascade Network for UAV-Based Vehicle Detection | 2026 | 10.1109/TITS.2026.3717450 | Phát hiện xe từ UAV (chưa có abstract; chỉ xếp dự phòng) |

**Nhận xét:**
- Dossier A1 **đạt** (≥ 3).
- Trong nhóm này chưa bài nào nghiên cứu **tần suất chạy detector** hoặc lịch tuần hoàn so với lịch thích nghi, nên khoảng trống G1/G2 vẫn mở.
- T2 và T3 là hai bài nên trích trong phần intro để neo framing ITS.

## TVT (Plan B): đạt 4 bài (yêu cầu ≥ 2)

| # | Title | Năm | DOI | Vì sao cùng dạng |
|---|---|---|---|---|
| V1 | MVDNet: UAV Based Multi-Modal Multi-Vehicle Anchor Free Detection | 2025 | 10.1109/TVT.2025.3580119 | Phát hiện nhiều xe từ ảnh UAV, nhấn mạnh tốc độ; có mật độ và che khuất |
| V2 | Trajectory Poisson Multi-Bernoulli Mixture Filter for Traffic Monitoring Using a Drone | 2024 | 10.1109/TVT.2023.3310742 | MOT giám sát giao thông bằng drone |
| V3 | Vehicular/Non-Vehicular Multi-Class Multi-Object Tracking in Drone-Based Aerial Scenes | 2024 | 10.1109/TVT.2023.3332132 | MOT từ drone; nêu **camera motion** và missed detection (đúng với P3) |
| V4 | Optimal Task Offloading and Trajectory Planning for Collaborative Video Analytics With UAV-Assisted Edge in Disaster Rescue | 2024 | 10.1109/TVT.2023.3344281 | Video analytics có ràng buộc năng lượng với UAV (khác dạng một phần, dùng làm đỡ framing năng lượng) |
