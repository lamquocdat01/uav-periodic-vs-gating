# P2D-C: chờ máy rảnh (CPU tổng < 30 % trung bình 60 s VÀ RAM trống >= 8 GB) rồi chạy benchmark weights thật.
# Đo mỗi 20 phút; tối đa 12 h. Mọi lần đo ghi vào results/p2/bench_real/idle_gate.log.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File code\p2\p2_wait_idle_bench.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$out = Join-Path $root "results\p2\bench_real"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "idle_gate.log"
$models = if ($env:THS_DATASETS) { $env:THS_DATASETS } else { "$THS_DATASETS" }
$ir = Join-Path $models "_models\28\yolo26s_visdrone_1024\export"
$irn = Join-Path $models "_models\28\yolo26n_visdrone_1024\export"   # P2E: yolo26n VisDrone (Kaggle v3); tiny@320 from ..\export_local
$deadline = (Get-Date).AddHours(12)
while ((Get-Date) -lt $deadline) {
    $s = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 12
    $cpu = ($s.CounterSamples | Measure-Object CookedValue -Average -Maximum)
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $line = "{0:yyyy-MM-dd HH:mm:ss} CPU avg {1:N1}% max {2:N1}% RAM free {3:N1} GB" -f (Get-Date), $cpu.Average, $cpu.Maximum, $free
    if ($cpu.Average -lt 30 -and $free -ge 8) {
        Add-Content $log "$line -> IDLE, start bench" -Encoding utf8
        Write-Output "$line -> IDLE, start bench"
        Set-Location $root
        # stderr của python (cảnh báo) không được coi là lỗi: chạy qua cmd, gộp 2>&1 bên trong cmd, ghi UTF-8
        $ErrorActionPreference = "Continue"
        $env:PYTHONIOENCODING = "utf-8"
        cmd /c ".venv\Scripts\python.exe code\p2_bench_openvino.py --real `"yolo26s=$ir`" `"yolo26n=$irn`" --sizes 640 960 1024 1280 --runs 100 --warmup 10 --out `"$out`" > `"$out\bench_stdout.log`" 2>&1"
        Add-Content $log ("{0:yyyy-MM-dd HH:mm:ss} bench exit {1}" -f (Get-Date), $LASTEXITCODE) -Encoding utf8
        exit $LASTEXITCODE
    }
    Add-Content $log "$line -> busy, wait 20 min" -Encoding utf8
    Write-Output "$line -> busy"
    Start-Sleep -Seconds 1140
}
Add-Content $log ("{0:yyyy-MM-dd HH:mm:ss} TIMEOUT 12 h, bench not run" -f (Get-Date)) -Encoding utf8
exit 9
