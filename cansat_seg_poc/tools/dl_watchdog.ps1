param(
    [int]$IdleSeconds = 300,
    [int]$EverySeconds = 60
)

# Watchdog de descargas v2.
#
# ⚠ La versión anterior leía (Get-Item).Length / LastWriteTime, que en NTFS
#   quedan CACHEADOS en la entrada de directorio: un archivo creciendo a 2 MB/s
#   puede parecer congelado durante minutos. Eso produjo falsos "estancados".
#   Acá el tamaño se lee abriendo el archivo (FileStream), que siempre ve el
#   tamaño real del kernel.
#
# Si un archivo no crece de verdad en $IdleSeconds, mata el curl que lo escribe;
# tools/dl_loop.ps1 lo relanza solo con -C - (resume).

$targets = @(
    @{ f = "C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\dataset\downloads\bright\bright_pre-event.zip"; s = 7977829095 },
    @{ f = "C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\dataset\downloads\bright\bright_post-event.zip"; s = 3291986721 },
    @{ f = "C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\dataset\downloads\rescuenet\rescuenet_segmentation-trainset.zip"; s = 18699171789 },
    @{ f = "C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\dataset\downloads\rescuenet\rescuenet_segmentation-validationset.zip"; s = 2373908074 }
)

$log = "C:\Users\manuy\PROYECTOS_Y_ESTUDIOS\CANSAT\cansat_seg_poc\outputs\dl_watchdog.log"
"$(Get-Date -Format s) watchdog v2 iniciado (idle=$IdleSeconds s)" |
    Out-File -Append -Encoding utf8 $log

function Get-RealLength([string]$path) {
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $len = $fs.Length
        $fs.Close()
        return $len
    } catch {
        return -1
    }
}

$lastLen = @{}
$lastChange = @{}

while ($true) {
    $now = Get-Date
    foreach ($t in $targets) {
        if (-not (Test-Path -LiteralPath $t.f)) { continue }
        $len = Get-RealLength $t.f
        if ($len -lt 0) { continue }
        if ($len -ge $t.s) { continue }

        if (-not $lastLen.ContainsKey($t.f) -or $len -ne $lastLen[$t.f]) {
            $lastLen[$t.f] = $len
            $lastChange[$t.f] = $now
            continue
        }

        $idle = ($now - $lastChange[$t.f]).TotalSeconds
        if ($idle -gt $IdleSeconds) {
            $killed = 0
            Get-CimInstance Win32_Process -Filter "Name='curl.exe'" |
                Where-Object { $_.CommandLine -like "*$($t.f)*" } |
                ForEach-Object {
                    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                    $killed++
                }
            if ($killed -gt 0) {
                "$(Get-Date -Format s) $([IO.Path]::GetFileName($t.f)): ${idle}s sin crecer de verdad, $killed curl matado(s)" |
                    Out-File -Append -Encoding utf8 $log
                $lastChange[$t.f] = $now
            }
        }
    }
    Start-Sleep -Seconds $EverySeconds
}
