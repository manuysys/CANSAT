# Guía de Despliegue del CANSAT - Exponiendo IA en Vivo con Raspberry Pi

## Resumen
Esta guía describe los pasos para desplegar la cámara en vivo con inferencia de IA en la Raspberry Pi Zero W.

## Requerimientos previos
La Pi debe estar EN ESTA RED (Salon-Wifi-2.4).

### Si la Pi usa WiFi viejo:
Editar `/boot/wpa_supplicant.conf` en la SD:
```
country=AR
ctrl_interface=Dir=/var/run/wpa_supplicant Group=netdev
update_config=1

network={
    ssid="Salon-Wifi-2.4"
    psk="TU_CONTRASEÑA"
    key_mgmt=WPA-PSK
}
```

### Si la Pi usa USB Ethernet:
Habilitar RNDIS en `/boot/config.txt`:
```
dtoverlay=dwc2
dtoverlay=rpi-proto
```

## Archivos del proyecto
```
cansat_seg_poc/pi/
├── ai_expo_vivo.py      - Servidor IA con cámara + ONNX
├── web_telemetry.py     - Dashboard telemetría LoRa RX
├── outputs/
│   ├── terrain_seg.onnx      - Segmentación de terreno (4 clases)
│   ├── damage_detection.onnx - Detección de daños (5 clases)
│   ├── fire_smoke.onnx       - Fuego y humo (3 clases)
│   ├── flood.onnx            - Inundación (1 clase)
│   └── flood_severity.onnx   - Severidad de inundación (3 clases)
└── models/
    └── coco_labels.txt   - Etiquetas COCO para detección
```

## Pasos de despliegue

### 1. Verificar conexiones
```bash
# Comprobar que estamos en la red correcta
ip addr
# Debería mostrar algo como 192.168.2.x

# Probar LoRa RX
cat /dev/ttyUSB0  # o ttyACM0
# Debería ver paquetes $LB135,x,x,... con RSSI en comentarios
```

### 2. Ejecutor del server
```bash
# Verificar Python y OpenCV
python3 --version
python3 -c "import cv2; print(cv2.__version__)"
# OpenCV 4.x es necesario para DNN (reemplaza onnxruntime)

# Verificar dispositivos USB
lsusb
# Debería ver el módulo WiFi y el módulo LoRa

# Verificar que tenemos la cámara
ls /dev/v4l/
# O dmesg | grep -i camera
```

### 3. Desplegar el server
```bash
cd ~/cansat_seg_poc
python3 ai_expo_vivo.py --model-outputs outputs/
# Server correrá en puerto 8050
# URL: http://<ip-de-la-pi>:8050
```

### 4. Acceder desde el navegador
Abrir `http://<ip-de-la-pi>:8050`

La interfaz muestra:
- Cámara en vivo streaming (MJPEG)
- Anotaciones de modelos ONNX sobre la imagen
- Bounding boxes y etiquetas con confidencias
- Panel lateral con historial de detecciones

## Archivos a copiar a la Pi

### ONNX models
Los 11 modelos están en `cansat_seg_poc/pi/outputs/`
Copiarlos con:
```bash
scp -r outputs/* pi@192.168.2.xx:~/cansat_seg_poc/outputs/
```

### Scripts
```bash
scp ai_expo_vivo.py pi@192.168.2.xx:~/cansat_seg_poc/
```

## Troubleshooting

### La cámara no aparece
```bash
lsusb | grep -i cam
dmesg | tail -20
# Instalar si no está:
sudo apt install python3-picamera2  # Pi Camera V2
```

### Error con OpenCV DNN
Si OpenCV no tiene módulo DNN disponible:
```bash
pip install opencv-contrib-python
```

### Servidor no se conecta al RX
Verificar que el LoRa está configurado con:
- SF9 (Spreading Factor 9)
- BW125 (125 kHz ancho de banda)
- Sync word 0xF3
- Power 17dBm
