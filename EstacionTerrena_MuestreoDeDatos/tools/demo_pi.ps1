# ═══════════════════════════════════════════════════════════════════════════
#  demo_pi.ps1 — orquesta la demo "Pi en vivo" (se corre EN la PC)
# ═══════════════════════════════════════════════════════════════════════════
#  1. La Pi captura en vivo (AI Camera + Heltec por UART)  → pi/demo_vivo.sh
#  2. Trae la misión por scp al repo de vuelo
#  3. Post-vuelo completo en la PC (onnxruntime; sin EDSR para que sea rápido)
#  4. Copia misión + entrega a la estación terrena
#  5. Deja la estación lista para abrir en http://localhost:8000
#
#  Uso:
#      powershell -ExecutionPolicy Bypass -File tools\demo_pi.ps1
#      powershell -ExecutionPolicy Bypass -File tools\demo_pi.ps1 -Frames 20
#      powershell -ExecutionPolicy Bypass -File tools\demo_pi.ps1 -Pi 192.168.68.240
param(
    [string]$Pi = "192.168.68.240",
    [int]$Frames = 12,
    [switch]$SaltarPostVuelo
)
$ErrorActionPreference = "Stop"

$Station = Split-Path -Parent $PSScriptRoot          # raíz de la estación
$Flight = Join-Path (Split-Path -Parent $Station) "cansat_seg_poc"
$SSH = @("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "pi@$Pi")
$SCP = @("scp", "-q", "-o", "BatchMode=yes")

Write-Host "1/5 Capturando $Frames frames en la Pi ($Pi)..." -ForegroundColor Cyan
& $SSH[0] $SSH[1..($SSH.Count-1)] "bash ~/cansat_seg_poc/pi/demo_vivo.sh $Frames"
if ($LASTEXITCODE -ne 0) { throw "la captura en la Pi falló (exit $LASTEXITCODE)" }
$src = (& $SSH[0] $SSH[1..($SSH.Count-1)] "ls -dt ~/vuelos/demo_* | head -1").Trim()
if (-not $src) { throw "no encontré la misión en la Pi" }
Write-Host "  misión en la Pi: $src"

Write-Host "2/5 Trayendo la misión..." -ForegroundColor Cyan
$mission = Join-Path $Flight "outputs\mission"
if (Test-Path $mission) { Remove-Item "$mission\*" -Recurse -Force }
New-Item -ItemType Directory -Force -Path $mission | Out-Null
& $SCP[0] $SCP[1..($SCP.Count-1)] -r "pi@${Pi}:$src/mission/*" $mission

if (-not $SaltarPostVuelo) {
    Write-Host "3/5 Post-vuelo completo en la PC (sin EDSR)..." -ForegroundColor Cyan
    Push-Location $Flight
    try {
        # En modo cámara los frames crudos van a full_res (high_res es del sampler).
        python post_flight.py --frames outputs\mission\full_res --n $Frames --no-edsr
        if ($LASTEXITCODE -ne 0) { throw "post_flight.py falló (exit $LASTEXITCODE)" }
    } finally { Pop-Location }
} else {
    Write-Host "3/5 Post-vuelo SALTEADO (-SaltarPostVuelo)" -ForegroundColor Yellow
}

Write-Host "4/5 Copiando misión + entrega a la estación..." -ForegroundColor Cyan
$stMission = Join-Path $Station "outputs\mission"
if (Test-Path $stMission) { Remove-Item "$stMission\*" -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stMission | Out-Null
Copy-Item "$mission\*" $stMission -Recurse -Force
$stEntrega = Join-Path $Station "entrega"
if (Test-Path $stEntrega) { Remove-Item "$stEntrega\*" -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stEntrega | Out-Null
if (Test-Path (Join-Path $Flight "entrega")) {
    Copy-Item (Join-Path $Flight "entrega\*") $stEntrega -Recurse -Force
}
$corr = Join-Path $Flight "outputs\corridor_map.jpg"
if (Test-Path $corr) { Copy-Item $corr (Join-Path $Station "outputs\corridor_map.jpg") -Force }

Write-Host "5/5 Listo. Estación:" -ForegroundColor Green
Write-Host "     cd $Station; python web_server.py    → http://localhost:8000" -ForegroundColor Green
