# P2G-D: chạy hàng đợi dump còn lại (CHỈ MÔ TẢ, không chặn freeze) ở chế độ ẩn, không gắn console.
# Giả thuyết (LOG_28, 01-10-2026): hai lần chết im lặng (30-09 21:25, 01-10 15:21) là do cửa sổ console riêng của queue
# (CREATE_NEW_CONSOLE) bị đóng / nhận Ctrl+C — cả cmd + powershell + python cùng biến mất, không có dòng "queue exit", .err rỗng.
# Chạy ĐÊM, SAU KHI anh Đạt pause Google Drive. Resume tự động (p2_dump bỏ qua chuỗi đã có parquet).
# Dùng:  powershell -NoProfile -ExecutionPolicy Bypass -File code\p2\p2_dump_queue_hidden.ps1 [-Steps yolo26s_960,...]
# Dừng sạch: New-Item $THS_DATASETS\_derived\28\dumps\_queue.STOP ; theo dõi: results\p2\dump_queue.log, results\p2\dump_queue_hidden.log
param([string[]]$Steps = @("yolo26s_960", "yolo26s_1280", "yolo26n_960", "yolo26n_1280"))
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$log = Join-Path $root "results\p2\dump_queue_hidden.log"
$queue = Join-Path $root "code\p2\p2_dump_queue.ps1"
$args_ = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$queue`"", "-Steps", ($Steps -join ","))
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args_ -WindowStyle Hidden -WorkingDirectory $root -PassThru `
     -RedirectStandardOutput $log -RedirectStandardError "$log.err"
"{0:yyyy-MM-dd HH:mm:ss} hidden queue PID {1} steps {2} (stdout {3})" -f (Get-Date), $p.Id, ($Steps -join ","), $log
