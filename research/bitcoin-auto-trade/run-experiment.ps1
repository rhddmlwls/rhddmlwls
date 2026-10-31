param(
    [string]$Config = "configs/btc.toml",
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw "Create a Python 3.12 .venv and install this project first."
    }
    $taskArguments = @("-m", "trading.run", "--config", $Config)
    if ($CheckOnly) { $taskArguments += "--check" }
    & $taskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
