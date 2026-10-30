# CANSAT Exposé: Camera + AI Live

## Overview
This demo shows a live camera stream from the Raspberry Pi Zero W running real-time object detection and semantic segmentation using ONNX models with OpenCV DNN.

## Architecture
```
[Pi Camera] --> [OpenCV DNN Inference] --> [HTTP MJPEG Stream] --> [Browser]
                    |                          |
              bounding boxes            Annotations
              confidence scores         with labels
```

## Models Available
- terrain_seg: Semantic segmentation of terrain types
- damage_detection: Damage assessment (5 classes)
- fire_smoke: Fire and smoke detection (3 classes)
- flood: Flood detection (1 class)
- flood_severity: Flood severity (3 classes)

## Deployment Steps (when PI is accessible)

### 1. Copy files to PI
```bash
cd ~/PROYECTOS_Y_ESTUDIOS/CANSAT
scp -r cansat_seg_poc/pi outputs ai_expo_vivo.py pi@192.168.2.IP:~/cansat_seg_poc/
```

### 2. Start server
```bash
ssh pi@192.168.2.IP
cd ~/cansat_seg_poc
python3 ai_expo_vivo.py --model-dir outputs/ --camera 0
```

### 3. Open browser
```
http://192.168.2.IP:8080
```

## Local Development
For testing the UI without the PI:
```bash
cd cansat_seg_poc/pi
python3 -m http.server 8080 --directory web/
```

## Notes
- OpenCV DNN is required (not onnxruntime) due to ARMv6 constraints
- Pi Zero W has 426MB RAM, optimized inference pipeline is essential
- Camera requires Picamera2 or USB camera module
- LoRa RX (if connected) provides telemetry for sensor fusion
