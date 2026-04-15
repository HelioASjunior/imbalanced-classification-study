$pythonExe = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Ambiente .venv nao encontrado. Crie com: py -3.12 -m venv .venv"
    exit 1
}

& $pythonExe -m src.main
