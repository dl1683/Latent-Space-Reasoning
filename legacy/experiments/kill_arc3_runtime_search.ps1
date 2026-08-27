$ErrorActionPreference = "Stop"

try {
    $matches = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*search_ls20_runtime.py*" }

    if (-not $matches) {
        Write-Host "No search_ls20_runtime.py python process found."
        exit 0
    }

    foreach ($process in $matches) {
        Write-Host "Stopping PID $($process.ProcessId): $($process.CommandLine)"
        Stop-Process -Id $process.ProcessId -Force
    }
} catch {
    Write-Host "PowerShell process lookup failed: $($_.Exception.Message)"
    Write-Host "Fallback: use Task Manager, or run this from an elevated terminal after identifying the exact PID:"
    Write-Host "taskkill /PID <pid> /F"
    exit 1
}
