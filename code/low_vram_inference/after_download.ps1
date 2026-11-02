param([int]$DownloadPid = 0, [switch]$SelfCheck)
$ErrorActionPreference = 'Stop'
if ($SelfCheck) {
    $ErrorActionPreference = 'Continue'
    $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -DownloadPid 0 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0 -or $output -notmatch 'not the experiment downloader') { throw 'PID guard check failed' }
    Write-Output 'Unrelated/missing PID rejected before launch'
    exit 0
}
$Root = 'D:\ai-research\low-vram-inference'
$download = Get-CimInstance Win32_Process -Filter "ProcessId=$DownloadPid"
if (!$download -or !$download.CommandLine -or !$download.CommandLine.Contains("$Root\download_deepseek.py")) {
    throw 'PID is not the experiment downloader'
}
[IO.File]::WriteAllText("$Root\results\after-download.pid", [string]$PID)
Write-Output "Waiting for verified download PID $DownloadPid; controller PID $PID"
Wait-Process -Id $DownloadPid
& 'C:\Users\marks\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\python.exe' "$Root\download_deepseek.py" --check-verified
if ($LASTEXITCODE -ne 0) { throw 'Complete publisher verification artifact unavailable; model launch refused' }
& 'C:\Users\marks\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\python.exe' "$Root\run_deepseek.py" --gpu-layers 0
if ($LASTEXITCODE -ne 0) { throw 'CPU experiment did not pass; inspect timestamped results' }
