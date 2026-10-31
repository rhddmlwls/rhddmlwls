param(
    [string]$Config = "configs/btc.toml",
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw "Project Python is missing. Create .venv with Python 3.12, then install this project with pip install -e .[plots]."
    }
    $taskArguments = @("-m", "trading.cli", "experiment", "--config", $Config)
    if ($CheckOnly) { $taskArguments += "--check" }
    & $taskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
