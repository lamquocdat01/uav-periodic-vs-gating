# P3-A4: hàng đợi TEST tuần tự (1 worker): dump yolo26s@1024 → dump yolo26n@640 → cue trace 4 cue (θ* đã đóng băng, không tính lại).
# Chỉ gọi code/p2 đã đóng băng: p2_dump.py --allow-test (GPU, RAM ≥ 6 GB) và p2_cue_trace.py --split test.
# Log: results/p3/logs/test_queue.log (PID, mốc giờ, exit) + stdout từng bước results/p3/logs/<step>.log
# Chạy ẩn: Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',<this>
$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$logdir = Join-Path $root "results\p3\logs"
New-Item -ItemType Directory -Force $logdir | Out-Null
$log = Join-Path $logdir "test_queue.log"
function Log($msg) { Add-Content $log ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg) -Encoding utf8 }
Log "queue start PID $PID"
$steps = @(
    @("dump_yolo26s_1024", @("code\p2\p2_dump.py", "--detector", "yolo26s", "--imgsz", "1024", "--split", "test", "--device", "GPU", "--max-workers", "1", "--min-free-ram-gb", "6", "--allow-test")),
    @("dump_yolo26n_640", @("code\p2\p2_dump.py", "--detector", "yolo26n", "--imgsz", "640", "--split", "test", "--device", "GPU", "--max-workers", "1", "--min-free-ram-gb", "6", "--allow-test")),
    @("cue_trace_test", @("code\p2\p2_cue_trace.py", "--dataset", "uavdt", "--split", "test", "--cues", "raw_diff,border_band,orb_lite_diff,tiny_det"))
)
foreach ($s in $steps) {
    $name, $a = $s[0], $s[1]
    $out = Join-Path $logdir "$name.log"
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    $null = $p.Handle   # giữ handle để đọc được ExitCode (lần chạy 15:28 trả ExitCode rỗng → queue dừng nhầm)
    Log "step $name child PID $($p.Id)"
    $p.WaitForExit()
    Log "step $name exit $($p.ExitCode)"
    if ($p.ExitCode -ne 0) { Log "queue stop (exit != 0)"; exit $p.ExitCode }
}
Log "queue done"
