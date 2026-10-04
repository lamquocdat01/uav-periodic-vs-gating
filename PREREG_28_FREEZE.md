# PREREG_28_FREEZE — đóng băng đăng ký trước, Đề tài 28

- Thời điểm: **2026-10-01 17:04:34 +0700**
- Commit nội dung được đóng băng (HEAD trước commit freeze): **`f743dfda93f6266eb07da90b5c63d5d5f18d254d`**; tag `prereg-28-v1` trỏ vào commit "prereg-28 freeze" chứa file này.
- Môi trường: ultralytics **8.4.163**, openvino **2026.4.0-22959-99c81491cc3-releases/2026/4**, Python 3.11.9, Windows-10-10.0.26200-SP0
- Quyết định đóng băng: anh Đạt, 01-10-2026. TEST chưa mở tại thời điểm này.

## Luật sau freeze

- TEST chỉ được mở qua `code/p2/p2_dump.py --allow-test` (dump) và `code/p2/p2_prereg_score.py` (chấm); audit TEST qua `p2_audit_prebirth.py --allow-test` (chỉ mô tả).
- Không sửa `code/p2/` trừ sửa lỗi: mỗi sửa ghi LOG_28 (lý do, diff, ảnh hưởng tới dự đoán) và tag lại `prereg-28-v1.1` (v1.2, …) kèm lý do.

## sha256 file được đóng băng

| file | sha256 | byte |
|---|---|---|
| `PREREG_28.md` | `81c19b43473e54d9be13be4e7ecaebbaab9e35d479f6027b247afc6b0a3640ab` | 24399 |
| `PREREG_28_DRAFT.md` | `67f3fcc548360be92756fef02f2031a2ebde8bc22b91b5ace456ee9917834645` | 17765 |
| `THEORY_28.md` | `f3a949964c171d04005b2d19dc67bdfb07e5a272ff911292a08422963ef2c15b` | 35996 |
| `results/p2/channel_train.json` | `1b37ac3037927f05c7d7918b2775841e8d7bdf2302fcc258d5611ad904c24d1e` | 104446 |
| `results/p2/prereg_cells.json` | `fb30790c2057755dfbc510e9001ef662a4ad3277b7934da2099a8fe41c8a087e` | 3533952 |
| `results/p2/detector_recall_train.json` | `f7ae533bad515eb92edf1ff76d583534200ecfe78c028a826a4d5ecf62325076` | 3672 |
| `results/p2/bench_real/bench_c_real.json` | `8c44c795481031b1c9d623e9e423feeb8db356c2a33738e4ec18767822a167a3` | 19544 |
| `code/p2/p2_audit_prebirth.py` | `272d160773e3aed438bb91f7d21ac15c1f9a7905dc03c5c68360709950e438fa` | 10680 |
| `code/p2/p2_bench_c_real.py` | `bd87409655f2ff6c56b40c02ae119b54162432a05258f3db7000b1ff16b1366a` | 6710 |
| `code/p2/p2_check_tex_numbers.py` | `70e0359d06d63677990e8f0b9006d8478378f22d7b0465324d63f048473a703b` | 9522 |
| `code/p2/p2_crosscheck_exports.py` | `ec0513323472f71074ebc5dd4743b0b69841b7a4aa5ae1e20b7bec3eb54f5da4` | 4786 |
| `code/p2/p2_cue_trace.py` | `0167cb7efd5ef428064ca4c32a20ff46aa17c96c9a38efee7870e3316b9e42f1` | 3057 |
| `code/p2/p2_cues.py` | `c660f5a36ab0952078134864d74624c5e03a5fc519c117fff14ee2f6ea2c5fea` | 7327 |
| `code/p2/p2_detector_recall.py` | `031f2fba12787603ef7fa63028597ab9304bdc89f202ec8a97e925c66d87ad9f` | 4603 |
| `code/p2/p2_dump.py` | `6bde657d29ce8a7f0ce08b390bedb189ed85133c46e1da186babb0f8ae8775e5` | 7967 |
| `code/p2/p2_dump_summary.py` | `81d9dba0e9ad76985696664adc7456b2ee831b3bb2d7a19d52572594c9fad92d` | 3097 |
| `code/p2/p2_estimate_channel.py` | `e684a52ec3aa847b299df3997b4f83734c886ecaf1d6cb16b0442f0d1b8ebe15` | 16729 |
| `code/p2/p2_export_local_reference.py` | `bcbdfd88cab7ada720322f831c417465c3b645180c852d92d7dba77321540558` | 2991 |
| `code/p2/p2_figures.py` | `86ed2347cc3c13fee04fe76988a117056305cd3042943bbbdcf46c47fde0dc44` | 20791 |
| `code/p2/p2_frames.py` | `e43013141c17b6546502f40e0eebd54473485c950891d5a8c96e7206138425c9` | 1009 |
| `code/p2/p2_numbers.py` | `2180f3ee1f5a9564a745ae180daec3a0d5915404d145071c6bee7402841bb13e` | 33862 |
| `code/p2/p2_prereg_build.py` | `3d64941ddcd92e200d5d0a59590de9fbba5fdd4703bf7f8f1906db89ff2ec59c` | 45202 |
| `code/p2/p2_prereg_score.py` | `54b85c0cad923d74593083f081ce811aec72690daf731251283d4ca772abd8a4` | 7219 |
| `code/p2/p2_prereg_simtest.py` | `4ca178f65dd7389317a3ab77863549d5cd73693308898a5fd53647bf6fd355f4` | 7910 |
| `code/p2/p2_scene_overlap.py` | `345fd7ccd053242616abf876f6bf1468370ad9073e9aadf8098f53ca6cd8a58e` | 4067 |
| `code/p2/p2_unzip_uavdt.py` | `a28ca8854a55fd5d65c6d9e2e5982ccaa691224cc305697caa6aca6b130217d5` | 4963 |
| `code/paths.py` | `218eb6dbce3394aa66103decf42db986e3019136b1fe70b4e9f1e82cc533f5ee` | 3752 |
| *ngoài lệnh freeze, thêm vì scorer/replay dùng trên TEST:* | | |
| `code/p2/replay/p2_dump_schema.py` | `ff0ead53d32107c1467c66cb059dca597c7d19083fd00328fc92dd20dd0db159` | 3273 |
| `code/p2/replay/p2_engine.py` | `0c7b5e20fb248d4956a443a3b1bb03d35475b65785d597a1dbeba07c49e512f0` | 12920 |
| `code/p2/replay/p2_match.py` | `417e3155dec138802aa095bdefd12ce2b2e778d22b6a70b79ef06159b9c60d7d` | 3831 |
| `code/p2/replay/p2_mstar_check.py` | `0cf2a8af964a47a0a31c5905fbc96cc95ae0d2770c2cd68484d927d1559c6104` | 8826 |
| `code/p2/replay/p2_mstar_iso.py` | `0ba464435f2af93e290364292a5e2bb4e15cdbc9a658d220b0394a6881dc42f2` | 8722 |
| `code/p2/replay/p2_multilook_check.py` | `1f8c73784e66d7489e0457d16b332e89b0c8c2c586f5a9889f1974964d5a55fc` | 3614 |
| `code/p2/replay/p2_sim_detector.py` | `16bc8e624d1932ba497c4bf4c04a02b2c65eb6ba77aa5d08f591ba4d5cc306b5` | 3603 |
| `code/p2/replay/p2_sim_report.py` | `33f5700f7e8049b8ff71072ca5c1101b7039d1e74e74c673f69129299738e203` | 15657 |
| `code/p2/replay/p2_sim_run.py` | `032cad9ec02485823db128e18a8b3e4e38812107cf641d18397937274647b742` | 18324 |
| `code/p2/replay/p2_stats.py` | `c8814b8939ab079cf0df406b1e2617310341e63b1d17e33c707fdcd3d2b3429e` | 3150 |
| `code/p2/replay/p2_tail_analysis.py` | `9ec5e09d83dc8840311441d420215e412ac3e486887a7c3053e36e9c345516d4` | 4830 |

## sha256 trọng số detector (ngoài repo, `$THS_DATASETS/_models/28/`)

| detector | best.pt | sha256 |
|---|---|---|
| yolo26s_visdrone_1024 (detector chính, Kaggle v2) | `yolo26s_visdrone_1024/weights/best.pt` | `825dffcc41f8bfcd871d97b1e48b8fc3ba8fad2002cd3c0e743ae07e5a57bf84` |
| yolo26n_visdrone_1024 (Kaggle v3; IR @640 = mức r thấp H8, @320 = cue tiny_det) | `yolo26n_visdrone_1024/weights/best.pt` | `4dc79ab7b900ee4ba382719e3c101ce8a56b6f045c3bbc9ae0af15d3b29472e3` |

Ghi chú: `bench_c_real.json` nằm ở `results/p2/bench_real/` (lệnh freeze ghi `results/p2/`, không có file ở đó).
