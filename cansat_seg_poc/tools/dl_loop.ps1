param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$Out,
    [Parameter(Mandatory=$true)][long]$Size
)

# Descarga reanudable en bucle: si curl se corta (red lenta, CDN), reintenta
# desde el byte donde quedó. Termina cuando el archivo alcanza el tamaño.
$dir = Split-Path -Parent $Out
New-Item -ItemType Directory -Force -Path $dir | Out-Null

while ($true) {
    $cur = 0
    if (Test-Path -LiteralPath $Out) { $cur = (Get-Item -LiteralPath $Out).Length }
    if ($cur -ge $Size) { break }
    & curl.exe -L -C - --connect-timeout 30 --speed-limit 20000 --speed-time 60 `
        --retry 5 --retry-delay 10 -o $Out $Url 2>$null
    Start-Sleep -Seconds 5
}
