param(
    [string]$Encoder = "Qwen/Qwen3-1.7B",
    [string]$Data = "..\\latent-space-encoding-evals\\gemini_training_merged.json",
    [string]$Cache = "checkpoints\\latent_scorer\\gemini_training_merged_latents.pt",
    [string]$Output = "checkpoints\\latent_scorer\\gemini_training_merged_judge.pt",
    [string]$InitCheckpoint = "checkpoints\\latent_scorer\\final_model.pt",
    [int]$LatentDim = 1024
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "experiments\\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "train_latent_scorer_watchdog.log"

$dataPath = $Data
$cachePath = $Cache
$outputPath = $Output
$initCheckpointPath = $InitCheckpoint
$latentDimValue = $LatentDim

$encodeBatch = 4
$maxLength = 2048
$attempt = 0

function Build-Args {
    param(
        [int]$EncodeBatch,
        [int]$MaxLength,
        [bool]$RebuildCache
    )

    $args = @(
        "experiments\\train_latent_scorer.py",
        "--data", $dataPath,
        "--encoder", $Encoder,
        "--quantization", "4bit",
        "--latent-dim", $latentDimValue,
        "--encode-batch-size", $EncodeBatch,
        "--max-length", $MaxLength,
        "--cache", $cachePath,
        "--output", $outputPath,
        "--init-checkpoint", $initCheckpointPath
    )
    if ($RebuildCache) {
        $args += "--rebuild-cache"
    }
    return $args
}

while ($true) {
    $attempt += 1
    $rebuildCache = -not (Test-Path $cachePath)

    "==== Run #$attempt $(Get-Date -Format s) ====" | Tee-Object -FilePath $logPath -Append
    $args = Build-Args -EncodeBatch $encodeBatch -MaxLength $maxLength -RebuildCache $rebuildCache
    $quotedArgs = $args | ForEach-Object { '"' + $_ + '"' }
    $cmdLine = "python " + ($quotedArgs -join " ")
    "cmd: $cmdLine" | Tee-Object -FilePath $logPath -Append

    cmd /c $cmdLine 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        "completed=1" | Tee-Object -FilePath $logPath -Append
        break
    }

    "exit_code=$exitCode" | Tee-Object -FilePath $logPath -Append

    $tail = Get-Content -Path $logPath -Tail 80 | Out-String
    if ($tail -match "out of memory" -or $tail -match "CUDA out of memory") {
        if ($encodeBatch -gt 1) {
            $encodeBatch = [Math]::Max(1, [int]($encodeBatch / 2))
            "adjustment=encode_batch_size new_value=$encodeBatch" | Tee-Object -FilePath $logPath -Append
        } elseif ($maxLength -gt 1024) {
            $maxLength = 1024
            "adjustment=max_length new_value=$maxLength" | Tee-Object -FilePath $logPath -Append
        }
    }

    Start-Sleep -Seconds 10
}
