# Run all v2 batches sequentially with GPU restart — self-logging
$projectDir = "C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning"
$logFile = "$projectDir\experiments\v2_run_log.txt"

Set-Location $projectDir
"Starting at $(Get-Date)" | Out-File $logFile

# Restart GPU
$devices = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NVIDIA*" -and $_.Class -eq "Display"}
foreach ($d in $devices) {
    "Restarting GPU: $($d.FriendlyName)" | Tee-Object -FilePath $logFile -Append
    Disable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false
    Start-Sleep -Seconds 3
    Enable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false
}
Start-Sleep -Seconds 10
"GPU reset done." | Tee-Object -FilePath $logFile -Append

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# Run batches sequentially (one model load at a time = safe VRAM)
$batches = @(
    @{name="A"; tasks="v2_01_ftc_unfairness,v2_02_gdpr_controller_processor,v2_03_employment_disparate_impact,v2_04_saas_contract_issues"; out="experiments/legal_v2_batch_a.json"},
    @{name="B"; tasks="v2_05_startup_acquisition_issues,v2_06_data_breach_risk_triage,v2_07_ip_risk_portfolio"; out="experiments/legal_v2_batch_b.json"},
    @{name="C"; tasks="v2_08_negotiation_leverage,v2_09_regulatory_response_strategy,v2_10_contractor_misclassification,v2_11_corporate_liability_shield,v2_12_whistleblower_retaliation"; out="experiments/legal_v2_batch_c.json"}
)

foreach ($b in $batches) {
    "===== BATCH $($b.name) starting at $(Get-Date) =====" | Tee-Object -FilePath $logFile -Append
    $cmd = "python experiments/run_legal_v2_comparison.py --model Qwen/Qwen3-4B --quantization 4bit --n-seeds 5 --max-new-tokens 2048 --skip-evolution --tasks $($b.tasks) --output $($b.out)"
    "Running: $cmd" | Tee-Object -FilePath $logFile -Append
    $result = & python experiments/run_legal_v2_comparison.py --model Qwen/Qwen3-4B --quantization 4bit --n-seeds 5 --max-new-tokens 2048 --skip-evolution --tasks $($b.tasks) --output $($b.out) 2>&1
    $result | Out-File -FilePath $logFile -Append
    "BATCH $($b.name) exit code: $LASTEXITCODE" | Tee-Object -FilePath $logFile -Append
}

"===== ALL DONE at $(Get-Date) =====" | Tee-Object -FilePath $logFile -Append
