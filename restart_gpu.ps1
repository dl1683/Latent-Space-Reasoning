$devices = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NVIDIA*" -and $_.Class -eq "Display"}
foreach ($d in $devices) {
    Write-Host "Found: $($d.FriendlyName) [$($d.Status)]"
    Disable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false
    Start-Sleep -Seconds 3
    Enable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false
    Write-Host "Re-enabled."
}
Start-Sleep -Seconds 5
Write-Host "GPU reset complete."
