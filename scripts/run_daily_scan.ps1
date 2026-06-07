# run_daily_scan.ps1 — PowerShell wrapper for Task Scheduler
# Update $TradingAppPath before scheduling.
param()

$TradingAppPath = "C:\Users\gordo\Documents\trading_app"   # UPDATE THIS PATH
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$LogDir = Join-Path $TradingAppPath "logs"
$LogFile = Join-Path $LogDir "scan_$timestamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

"[$timestamp] Starting daily scan" | Tee-Object -FilePath $LogFile
Set-Location $TradingAppPath

& python research\daily_scan.py 2>&1 | Tee-Object -FilePath $LogFile -Append

$exit = $LASTEXITCODE
"[$(Get-Date -Format 'HH:mm:ss')] Scan exited with code $exit" |
    Tee-Object -FilePath $LogFile -Append

exit $exit
