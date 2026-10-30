# Red por USB (NCM) — usar la Pi Zero sin WiFi

**Fecha:** 2026-10-30. Verificado en banco con una **Raspberry Pi Zero W v1**
(Raspbian 13 Trixie, hostname `cansat`).

## Para qué sirve

Permite usar la Pi en cualquier lugar **sin depender de una red WiFi**: el
puerto **USB/OTG** se presenta como una placa de red (USB NCM). La PC le asigna
`192.168.7.1/24`, la Pi queda en `192.168.7.2/24` y se entra por SSH con un solo
cable. Ese mismo cable alimenta la Pi.

## Estado verificado (2026-10-30)

| Ítem | Valor |
|---|---|
| Servicio | `usb-ncm-gadget.service` — `enabled`, `active (exited)` |
| Script | `/usr/local/sbin/usb-ncm-gadget.sh` |
| Gadget configfs | `/sys/kernel/config/usb_gadget/cansat` |
| VID:PID | `2e8a:0014` (Raspberry Pi) |
| Producto | `Pi Zero NCM` |
| Interfaz en la Pi | `usb0` **UP** `192.168.7.2/24` |
| Host (Windows) | `Ethernet 2` / `UsbNcm Host Device`, `192.168.7.1/24` |
| Prueba | `ping 192.168.7.2` OK · `ssh pi@192.168.7.2` → `cansat` |
| WiFi | `wlan0` conectado a `Matute deco` (`192.168.68.59/22`), NTP OK |

Log del servicio:

```text
NCM listo: UDC=20980000.usb DEV=usb0
```

> La Pi no tiene NTP cuando está sin WiFi: la hora del log puede estar corrida.
> No afecta el enlace.

## Configuración en la Pi

### 1. Device tree y módulos

`/boot/firmware/config.txt` (la SD actual tiene líneas viejas de otros ensayos):

```text
otg_mode=1
dtoverlay=dwc2,dr_mode=host
dtoverlay=dwc2
```

La **última** línea (`dtoverlay=dwc2`, sin `dr_mode`) es la que deja el puerto
en modo gadget; las dos anteriores son restos de ensayos de modo host. Para una
instalación limpia alcanza con:

```text
dtoverlay=dwc2
```

`/boot/firmware/cmdline.txt` (una sola línea):

```text
... rootwait modules-load=dwc2 ...
```

> No borrar `otg_mode=1` ni `dtoverlay=dwc2,dr_mode=host` de la SD actual sin
> probar: la combinación verificada es la que está. La limpieza se hace con
> backup y reinicio.

### 2. Script del gadget

`/usr/local/sbin/usb-ncm-gadget.sh` (root, `0755`):

```sh
#!/bin/sh
set -e
modprobe libcomposite
modprobe usb_f_ncm
G=/sys/kernel/config/usb_gadget/cansat
mkdir -p "$G"
echo 0x2e8a > "$G/idVendor"
echo 0x0014 > "$G/idProduct"
echo 0x0100 > "$G/bcdDevice"
echo 0x0200 > "$G/bcdUSB"
mkdir -p "$G/strings/0x409"
echo CANSAT > "$G/strings/0x409/manufacturer"
echo "Pi Zero NCM" > "$G/strings/0x409/product"
echo 0001 > "$G/strings/0x409/serialnumber"
mkdir -p "$G/configs/c.1/strings/0x409"
echo NCM > "$G/configs/c.1/strings/0x409/configuration"
mkdir -p "$G/functions/ncm.usb0"
[ -L "$G/configs/c.1/ncm.usb0" ] || ln -s "$G/functions/ncm.usb0" "$G/configs/c.1/"
for i in $(seq 1 120); do
  [ -n "$(ls /sys/class/udc 2>/dev/null)" ] && break
  sleep 1
done
UDC=$(ls /sys/class/udc | head -n1)
echo "$UDC" > "$G/UDC"
for i in $(seq 1 30); do
  DEV=$(ip -br link | awk '$1 ~ /^usb/ {print $1; exit}')
  [ -n "$DEV" ] && break
  sleep 1
done
[ -n "$DEV" ] || DEV=usb0
ip addr add 192.168.7.2/24 dev "$DEV" 2>/dev/null || true
ip link set "$DEV" up
echo "NCM listo: UDC=$UDC DEV=$DEV"
```

### 3. Servicio systemd

`/etc/systemd/system/usb-ncm-gadget.service`:

```ini
[Unit]
Description=USB NCM Ethernet gadget (192.168.7.2)
After=systemd-modules-load.service
Before=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/usb-ncm-gadget.sh

[Install]
WantedBy=multi-user.target
```

Instalación/reparación:

```bash
sudo install -m 755 usb-ncm-gadget.sh /usr/local/sbin/usb-ncm-gadget.sh
sudo systemctl daemon-reload
sudo systemctl enable --now usb-ncm-gadget.service
systemctl status usb-ncm-gadget.service
ip -brief addr show usb0
```

### 4. Perfil NetworkManager (opcional)

Existe `/etc/NetworkManager/system-connections/usb0-static.nmconnection`
(`manual`, `192.168.7.2/24`), pero la interfaz figura **unmanaged** y la IP
efectiva la pone el script del servicio. El perfil es redundante; no molesta.
Si en el futuro NetworkManager gestiona `usb0`, aplicaría la misma IP.

## Lado PC

### Windows 10/11

1. Conectar el cable al puerto **USB/OTG** de la Pi (el que está cerca del HDMI;
   **no** `PWR IN`). Ese puerto alimenta la Pi.
2. Aparece una placa nueva: `UsbNcm Host Device` (puede llamarse `Ethernet 2`,
   `Ethernet 3`, etc.).
3. Asignarle IP fija (requiere permisos de administrador):
   - IP: `192.168.7.1`
   - Máscara: `255.255.255.0`
   - Gateway: vacío
4. Probar:

```powershell
ping 192.168.7.2
ssh pi@192.168.7.2
```

No hay DHCP en el enlace: la IP del host se configura a mano. La primera vez
SSH pide aceptar la huella del host.

### Linux

```bash
ip -brief addr                 # identificar la interfaz nueva (usb0 / enx...)
sudo ip addr add 192.168.7.1/24 dev <interfaz>
sudo ip link set <interfaz> up
ping 192.168.7.2
ssh pi@192.168.7.2
```

### macOS

La interfaz NCM aparece en **Red**; asignarle `192.168.7.1/24` (sin router) y
entrar por SSH a `192.168.7.2`.

## Apagado seguro

La Pi se alimenta por el mismo cable USB. **No desenchufar en caliente:**

```bash
sudo shutdown -h now
```

Esperar a que el LED ACT deje de parpadear (unos segundos) y recién ahí
desconectar. Un corte en escritura puede corromper la microSD.

## WiFi (opcional, ya configurado)

Al 2026-10-30 la Pi tiene el perfil `Matute deco` guardado y `wlan0` conectado
(`192.168.68.59/22`, NTP sincronizado). El hardware WiFi está bien: escanea
`Matute deco` en 2,4 GHz (WPA2/WPA3, canales 2/4/10). Si se prepara otra SD y
falta el perfil:

```bash
sudo nmcli --ask device wifi connect "Matute deco"
```

Escribir la clave cuando la pida; el perfil queda guardado y se reconecta solo.
El gadget USB no necesita WiFi; la WiFi sirve para Internet/NTP/apt en casa.
En el CDR no depender del WiFi del lugar: la telemetría va por LoRa.

## Diagnóstico rápido

| Síntoma | Qué mirar |
|---|---|
| Aparece un **COM** y no una placa de red | El gadget es serie (`usb_f_acm`), no NCM. `lsmod \| grep usb_f_`; usar este servicio NCM |
| No aparece ninguna interfaz | Cable de datos y puerto **USB/OTG** (no `PWR IN`); `lsusb` en Linux debe mostrar `2e8a:0014`; en Windows, Administrador de dispositivos |
| `usb0` sin IP | `systemctl status usb-ncm-gadget.service`; `journalctl -u usb-ncm-gadget.service`; `ls /sys/class/udc`; reiniciar el servicio |
| La PC tiene `169.254.x.x` | Falta la IP fija `192.168.7.1/24`; no hay DHCP |
| SSH no conecta pero ping sí | `sshd` en la Pi; clave autorizada; firewall del host |
| La Pi no arranca el gadget | `dtoverlay=dwc2` en `config.txt`; `modules-load=dwc2` en `cmdline.txt`; backup y reinicio |

## Notas y límites

- Enlace **punto a punto**; no comparte Internet por defecto. La PC puede seguir
  usando su WiFi para Internet y el USB solo para `192.168.7.0/24`.
- El puerto USB de la Pi queda ocupado por el gadget: no conectar a la vez un
  teclado/host USB en ese puerto.
- El gadget usa VID/PID `2e8a:0014` y producto `Pi Zero NCM`; el COM anterior
  (`2e8a:0013`) era el gadget serie, ya reemplazado.
- En la SD actual quedaron `otg_mode=1` y `dtoverlay=dwc2,dr_mode=host` de
  ensayos previos. La línea efectiva es el último `dtoverlay=dwc2`. Limpiarlas
  solo con backup y prueba de reinicio.
