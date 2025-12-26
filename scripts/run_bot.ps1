param(
  [string]$Config = "configs/default.yaml",
  [string]$PythonPath = ".\\venv\\Scripts\\python.exe",
  [int]$RestartDelaySeconds = 5
)

# Simple watchdog to auto-restart the bot if it exits.
# Usage: .\scripts\run_bot.ps1 -Config configs/default.yaml

while ($true) {
  Write-Host "$(Get-Date -Format s) Starting bot with config=$Config"
  & $PythonPath -m div_donchian_bot.cli --config $Config
  $rc = $LASTEXITCODE
  Write-Warning "$(Get-Date -Format s) Bot exited with code=$rc. Restarting in $RestartDelaySeconds s..."
  Start-Sleep -Seconds $RestartDelaySeconds
}
