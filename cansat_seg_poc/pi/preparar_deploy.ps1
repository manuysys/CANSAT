# ═══════════════════════════════════════════════════════════════════════════
#  preparar_deploy.ps1 — arma dist_pi/ con lo mínimo para la Raspberry Pi
# ═══════════════════════════════════════════════════════════════════════════
#  Corre en la PC (Windows, PowerShell). NO toca los originales de outputs/:
#   · genera los modelos AUTOCONTENIDOS (sin .onnx.data) con tools/onnx_inline.py
#     porque OpenCV DNN (backend de la Pi Zero v1) no lee pesos externos;
#   · copia el paquete cansat/, los scripts de vuelo y pi/;
#   · deja listo el comando scp para copiar a la Pi.
#
#  Uso:
#     powershell -ExecutionPolicy Bypass -File pi\preparar_deploy.ps1
#     powershell -ExecutionPolicy Bypass -File pi\preparar_deploy.ps1 -Todos
#     powershell -ExecutionPolicy Bypass -File pi\preparar_deploy.ps1 -Destino C:\temp\pi
# ═══════════════════════════════════════════════════════════════════════════
param(
    [switch]$Todos,                # incluir TODOS los modelos (daño, flood, siamés)
    [string]$Destino = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot          # raíz del proyecto
if (-not (Test-Path (Join-Path $Root "mission_pipeline.py"))) {
    throw "No encuentro mission_pipeline.py en $Root. Corré el script desde el repo."
}
$Dist = if ($Destino) { $Destino } else { Join-Path $Root "dist_pi" }

Write-Host "══ Deploy para Raspberry Pi ══" -ForegroundColor Cyan
Write-Host "  proyecto : $Root"
Write-Host "  destino  : $Dist"

# ── 1. Modelos autocontenidos ────────────────────────────────────────────────
# Por defecto van los de VUELO: terreno a 224 px (mitad de cómputo que 320) y
# flood a 224. Con -Todos se agregan el terreno de 320, el tiny (NPU) y los
# modelos de daño/siamés (sólo útiles si la placa mide bien; ver guia_pi.md §6).
# El two-stage de vuelo es el adaptado a UAV con RescueNet (F2 2026-09-18).
$Modelos = @("cansat_seg_terrain_v2_224.onnx", "cansat_flood_specialist_224.onnx")
if ($Todos) {
    $Modelos += @(
        "cansat_seg_terrain_v2.onnx",
        "cansat_seg_terrain_tiny_224.onnx",
        "cansat_damage3_mobilenetv2.onnx",
        "cansat_damage_v3_bal.onnx",
        "cansat_fire_smoke.onnx",
        "cansat_severity.onnx",
        "cansat_siamese_damage.onnx"
    )
}
$ModelosExistentes = @()
foreach ($m in $Modelos) {
    if (Test-Path (Join-Path $Root "outputs\$m")) { $ModelosExistentes += "outputs\$m" }
    else { Write-Host "  [WARN] no existe outputs\$m (se saltea)" -ForegroundColor Yellow }
}
if (-not $ModelosExistentes) { throw "No hay ningún modelo para empaquetar en outputs/." }

Write-Host "`n── Empotrando pesos externos ──" -ForegroundColor Cyan
$OutModelos = Join-Path $Dist "outputs"
New-Item -ItemType Directory -Force -Path $OutModelos | Out-Null
Push-Location $Root
try {
    & python tools\onnx_inline.py @ModelosExistentes --out-dir $OutModelos
    if ($LASTEXITCODE -ne 0) { throw "onnx_inline.py falló (exit $LASTEXITCODE)" }
} finally { Pop-Location }

# ── 2. Código ────────────────────────────────────────────────────────────────
Write-Host "`n── Copiando código ──" -ForegroundColor Cyan
$Archivos = @(
    "mission_pipeline.py", "adaptive_sampler.py", "inference.py",
    "uart_listener.py", "sim_uart.py", "requirements-flight.txt"
)
foreach ($f in $Archivos) {
    Copy-Item (Join-Path $Root $f) -Destination $Dist -Force
}
Copy-Item (Join-Path $Root "cansat") -Destination $Dist -Recurse -Force
Copy-Item (Join-Path $Root "pi")     -Destination $Dist -Recurse -Force
# Herramienta de medición en la placa (ms por modelo) → dist_pi/tools/.
$ToolsDst = Join-Path $Dist "tools"
New-Item -ItemType Directory -Force -Path $ToolsDst | Out-Null
Copy-Item (Join-Path $Root "tools\bench_models.py") -Destination $ToolsDst -Force
# Referencia OOD (cansat/ood.py la busca en docs/benchmarks/): sin ella el
# aviso de fuera-de-distribución queda desactivado a bordo.
$OodRef = Join-Path $Root "docs\benchmarks\ood_loveda_val.json"
if (Test-Path $OodRef) {
    $OodDst = Join-Path $Dist "docs\benchmarks"
    New-Item -ItemType Directory -Force -Path $OodDst | Out-Null
    Copy-Item $OodRef -Destination $OodDst -Force
    Write-Host "  referencia OOD copiada (docs/benchmarks)"
}
# Datos auxiliares: grilla de densidad poblacional (WorldPop) para casualties.
$PopGrid = Join-Path $Root "dataset\population\population_grid.csv"
if (Test-Path $PopGrid) {
    $PopDst = Join-Path $Dist "dataset\population"
    New-Item -ItemType Directory -Force -Path $PopDst | Out-Null
    Copy-Item $PopGrid -Destination $PopDst -Force
    Write-Host "  grilla poblacional copiada (dataset/population, WorldPop CC BY 4.0)" -ForegroundColor DarkGray
}
# Limpiar cachés de Python que se cuelan con -Recurse.
Get-ChildItem -Path $Dist -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ── 3. Baseline (opcional, para el siamés) ───────────────────────────────────
$Baseline = Join-Path $Root "outputs\baseline.png"
if (Test-Path $Baseline) {
    Copy-Item $Baseline -Destination $OutModelos -Force
    Write-Host "  baseline.png copiado (siamés habilitado)"
} else {
    Write-Host "  [i] sin outputs/baseline.png: el siamés no se usará" -ForegroundColor DarkGray
}

# ── 3b. Tiles de prueba (smoke y bench SIN cámara en la Pi) ──────────────────
#  La Pi vuela con --folder cuando no hay cámara; el bench (tools/bench_models)
#  también necesita una imagen. Van 2–3 livianas de dataset/pruebas.
$Pruebas = Join-Path $Root "dataset\pruebas"
if (Test-Path $Pruebas) {
    $TilesDst = Join-Path $Dist "tiles"
    New-Item -ItemType Directory -Force -Path $TilesDst | Out-Null
    Get-ChildItem $Pruebas -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' } |
        Sort-Object Length | Select-Object -First 3 |
        ForEach-Object { Copy-Item $_.FullName -Destination $TilesDst -Force }
    $n = (Get-ChildItem $TilesDst -File).Count
    Write-Host "  $n imagenes de prueba copiadas a tiles/ (smoke y bench sin camara)"
} else {
    Write-Host "  [i] sin dataset/pruebas: no hay tiles para el smoke sin camara" -ForegroundColor DarkGray
}

# ── 4. Resumen ───────────────────────────────────────────────────────────────
$Peso = (Get-ChildItem $Dist -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n✔ dist_pi listo: $([math]::Round($Peso,1)) MB en $Dist" -ForegroundColor Green
Write-Host "  Contenido:"
Get-ChildItem $Dist | ForEach-Object { Write-Host "    $($_.Name)" }
Write-Host "`nCopiar a la Pi (ajustá usuario/IP):" -ForegroundColor Cyan
Write-Host "  scp -r `"$Dist\*`" pi@cansat.local:/home/pi/cansat_seg_poc/"
Write-Host "`nEn la Pi, después de copiar:"
Write-Host "  cd ~/cansat_seg_poc && bash pi/instalar_en_pi.sh"
