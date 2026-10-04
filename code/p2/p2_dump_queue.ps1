# P2E-C: sequential detector-dump queue on UAVDT TRAIN (GPU, 1 worker, RAM gate 6 GB), with resume.
# Order (PROMPT P2E C1-C5): yolo26s@1024 -> yolo26n@1024 -> yolo26s@640 -> yolo26s@960 -> yolo26s@1280
#                           -> yolo26n@640 -> yolo26n@960 -> yolo26n@1280 (C5 = optional, only if time).
# H8 core needs only (s@1024, n@1024) and (s@1024, s@640) = the first 3 steps.
# Resume: p2_dump.py skips sequences whose parquet already exists -> just start the queue again.
# Guards: refuses to start if (a) another queue holds the lock, (b) another p2_dump.py / p2_bench_openvino.py is running,
#         (c) UAVDT frames are missing. p2_dump exit 4 (free RAM < 6 GB) stops the queue (re-run later to resume).
# Stop:   create file  <DUMPS_DIR>\_queue.STOP  -> queue stops after the current step (clean);
#         immediate: Stop-Process -Id <child PID from dump_queue.log>  -> at most the current sequence is lost (resume redoes it).
# Log:    results\p2\dump_queue.log (PIDs, per-step frames / t_ms / boxes per frame), per-step stdout in results\p2\dump_logs\.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File code\p2\p2_dump_queue.ps1 [-Steps yolo26s_1024,yolo26n_1024] [-DryRun]
param(
    [string[]]$Steps = @("yolo26s_1024", "yolo26n_1024", "yolo26s_640", "yolo26s_960", "yolo26s_1280",
                         "yolo26n_640", "yolo26n_960", "yolo26n_1280"),
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$Steps = @($Steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File truyền "a,b" thành 1 chuỗi
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8   # read python stdout as UTF-8 (log lines had mojibake)
$datasets = if ($env:THS_DATASETS) { $env:THS_DATASETS } else { "$THS_DATASETS" }
$dumps = Join-Path $datasets "_derived\28\dumps"
$frames = Join-Path $datasets "UAVDT\frames\UAV-benchmark-M"
$log = Join-Path $root "results\p2\dump_queue.log"
$logdir = Join-Path $root "results\p2\dump_logs"
New-Item -ItemType Directory -Force $dumps, $logdir | Out-Null
$lock = Join-Path $dumps "_queue.lock"
$stop = Join-Path $dumps "_queue.STOP"
$py = Join-Path $root ".venv\Scripts\python.exe"

function Log($msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg
    # retry: log file may be held open by another reader (e.g. tail -F) -> retry up to 10 times
    for ($i = 0; $i -lt 10; $i++) { try { Add-Content $log $line -Encoding utf8 -ErrorAction Stop; break } catch { Start-Sleep -Milliseconds 500 } }
    Write-Output $line
}

# (a) lock
if (Test-Path $lock) {
    $old = (Get-Content $lock -TotalCount 1).Trim()
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) { Log "REFUSE: queue PID $old already running"; exit 5 }
    Log "stale lock (PID $old not alive) -> removed"
    Remove-Item $lock -Force
}
# (b) no other detector worker
# (psutil, not WMI: Win32_Process queries hung for minutes under load on 2026-09-30)
$busy = & $py -c "import psutil; print(','.join(str(p.pid) for p in psutil.process_iter(['cmdline']) if 'psutil' not in ' '.join(p.info['cmdline'] or []) and any(k in ' '.join(p.info['cmdline'] or []) for k in ('p2_dump.py', 'p2_bench_openvino.py'))))"
if ($busy) { Log "REFUSE: detector already running: PID $busy"; exit 5 }
# (c) frames
$nimg = if (Test-Path $frames) { (Get-ChildItem $frames -Recurse -Filter *.jpg -ErrorAction SilentlyContinue | Select-Object -First 1000).Count } else { 0 }
if ($nimg -eq 0) {
    if (-not $DryRun) { Log "REFUSE: no UAVDT frames in $frames (download UAV-benchmark-M.zip first)"; exit 6 }
    Log "DRYRUN: no UAVDT frames in $frames (a real run would refuse, exit 6)"
}

Set-Content $lock $PID -Encoding ascii
Log "QUEUE START PID $PID steps: $($Steps -join ' ')  (stop: New-Item '$stop'; or Stop-Process -Id <child>)"
$rc = 0
try {
    foreach ($st in $Steps) {
        if (Test-Path $stop) { Log "STOP file found -> queue stops before $st (delete $stop to allow restart)"; $rc = 7; break }
        $det, $sz = $st.Split("_")
        $args_ = @("code\p2\p2_dump.py", "--detector", $det, "--imgsz", $sz, "--split", "train", "--device", "GPU",
                   "--max-workers", "1", "--min-free-ram-gb", "6")
        if ($DryRun) { Log "DRYRUN $st : $py $($args_ -join ' ')"; continue }
        $out = Join-Path $logdir "$st.log"
        $t0 = Get-Date
        $p = Start-Process -FilePath $py -ArgumentList $args_ -NoNewWindow -PassThru `
             -RedirectStandardOutput $out -RedirectStandardError "$out.err"
        $null = $p.Handle   # cache handle now, otherwise Start-Process -PassThru ExitCode is empty
        Log "STEP $st start child PID $($p.Id)  (stdout $out)"
        $p.WaitForExit()
        $code = $p.ExitCode
        $h = ((Get-Date) - $t0).TotalHours
        $sum = & $py code\p2\p2_dump_summary.py --names $st 2>$null
        Log ("STEP $st exit $code, {0:N2} h: $sum" -f $h)
        if ($code -eq 4) { Log "RAM < 6 GB -> queue stopped; re-run the queue later to resume"; $rc = 4; break }
        if ($code -ne 0) { Log "p2_dump failed (exit $code) -> queue stopped; see $out.err"; $rc = $code; break }
    }
} finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
    Log "QUEUE END rc $rc"
}
exit $rc
