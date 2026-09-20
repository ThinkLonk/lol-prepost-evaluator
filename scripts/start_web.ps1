param(
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Missing .venv. Follow docs/React_FastAPI.md to install the environment.'
}
Push-Location $projectDirectory
try {
    & $pythonPath -X utf8 -c 'import fastapi, uvicorn'
    if ($LASTEXITCODE -ne 0) { throw 'Install requirements-web.txt before starting.' }
    if (-not $SkipBuild) {
        Push-Location (Join-Path $projectDirectory 'frontend')
        try {
            if (-not (Test-Path -LiteralPath 'node_modules')) {
                & npm.cmd ci --no-audit --no-fund
                if ($LASTEXITCODE -ne 0) { throw 'Failed to install frontend dependencies.' }
            }
            & npm.cmd run build
            if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
        } finally { Pop-Location }
    }
    if (-not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
        throw 'React build is missing. Run again without -SkipBuild.'
    }
    Write-Host "Match Insight: http://127.0.0.1:$Port"
    Write-Host "API: http://127.0.0.1:$Port/docs - Press Ctrl+C to stop."
    & $pythonPath -X utf8 -m uvicorn match_insight.api.app:app --host 127.0.0.1 --port $Port --workers 1
    if ($LASTEXITCODE -ne 0) { throw 'Server stopped with an error. See the output above.' }
} finally { Pop-Location }
