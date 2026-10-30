#!/usr/bin/env python3
"""
Servidor multi-modelo ONNX para la expo del CANSAT.
- Sirve cámara en vivo por HTTP (MJPEG stream)
- Ejecuta inferencia ONNX sobre frames de la cámara
- Soporta múltiples modelos: terrain segmentation, damage detection, fire/smoke
- Anota bounding boxes + labels sobre la imagen en vivo
"""
from __future__ import annotations

import argparse
import io
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import onnxruntime as ort

# --- Model loader ---

class Model:
    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = path.split("/")[-1].replace(".onnx", "")
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        print(f"  {self.name}: loaded, input={self.input_shape}")

    def run(self, img: np.ndarray) -> dict:
        inputs = {self.input_name: img.astype(np.float32)}
        outputs = self.sess.run(None, inputs)
        return self.postprocess(outputs)

    def postprocess(self, outputs):
        return {"raw": outputs, "frame": time.time()}


def load_models(args):
    models = {}
    model_paths = {
        "seg_tiny": "outputs/cansat_seg_terrain_tiny_224.onnx",
        "seg_tiny_int8": "outputs/cansat_seg_terrain_tiny_224_int8_qdq.onnx",
        "damage": "outputs/cansat_damage3_mobilenetv2.onnx",
        "fire_smoke": "outputs/cansat_fire_smoke.onnx",
    }

    if args.segment:
        for name, path in [("seg_tiny", model_paths["seg_tiny"]),
                           ("seg_tiny_int8", model_paths["seg_tiny_int8"])]:
            if args.int8 and name == "seg_tiny_int8":
                models[name] = Model(path)
    elif args.damage:
        if "outputs/cansat_damage3_mobilenetv2.onnx" in paths:
            models["damage"] = Model("outputs/cansat_damage3_mobilenetv2.onnx")
    elif args.fire:
        if "outputs/cansat_fire_smoke.onnx" in paths:
            models["fire_smoke"] = Model("outputs/cansat_fire_smoke.onnx")
    elif args.all:
        for name, path in model_paths.items():
            models[name] = Model(path)
    
    return models


# --- Color palette ---
TERRAIN_COLORS = {
    "vegetation": (0, 128, 0),
    "ground": (139, 69, 19),
    "water": (0, 0, 255),
    "concrete": (128, 0, 128),
    "road": (128, 128, 128),
}

DAMAGE_COLORS = {
    "no_damage": (0, 255, 0),
    "minor_damage": (255, 255, 0),
    "severe_damage": (0, 0, 255),
    "destroyed": (255, 0, 0),
}

FIRE_SMOKE_COLORS = {
    "background": (0, 0, 0),
    "fire": (0, 0, 255),  
    "smoke": (255, 255, 255),
}


def annotate_frame(frame: np.ndarray, models: dict, img_size: int, shutter: int) -> tuple:
    """Run inference on frame and annotate."""
    results = {}
    annotated = frame.copy()
    
    for name in models:
        # Resize for model
        small = cv2.resize(frame, (img_size, img_size))
        # BGR -> RGB -> NCHW
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img_input = img_rgb.transpose(2, 0, 1).astype(np.float32)
        
        if "seg" in name:
            # Segmentation model
            img_input = (img_input / 255.0)[:, np.newaxis, :, :]
            inputs = {models[name].input_name: img_input}
            try:
                outputs = models[name].sess.run(None, inputs)
                pred = outputs[0]  # (1, 4, H, W) or (1, 5, H, W) for damage
                if pred.shape[1] >= 5:
                    # Damage classification
                    damage_pred = pred[0, 4:]  # 4 classes
                    max_dmg = np.argmax(damage_pred)
                    max_val = damage_pred[max_dmg]
                    labels = {0: "no_damage", 1: "minor_damage", 
                              2: "severe_damage", 3: "destroyed"}
                    cls_name = labels.get(max_dmg, "unknown")
                    conf = float(max_val)
                    results[name] = {"type": "damage", "class": cls_name, "conf": conf}
                    color = DAMAGE_COLORS.get(cls_name, (255, 255, 255))
                    cv2.putText(annotated, f"{name}: {cls_name} {conf:.1%}",
                                (5, 30 + len(models) * 35 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Terrain segmentation: argmax over channels
                    pred_cls = pred[0].argmax(axis=0)  # (H, W)
                    unique_cls = np.unique(pred_cls)
                    results[name] = {"type": "seg", "classes": unique_cls.tolist(), "pred": pred_cls}
                    
                    # Overlay segmentation with transparency
                    overlay = np.zeros_like(annotated)
                    for ci, uv in enumerate(unique_cls):
                        if uv < len(TERRAIN_COLORS):
                            overlay[pred_cls == uv] = TERRAIN_COLORS[list(TERRAIN_COLORS.keys())[uv]]
                    alpha = 0.3
                    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
                    annotated = cv2.putText(annotated, f"{name}: {'|'.join(map(str, unique_cls))}",
                                            (5, 30 + len(models) * 35 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                results[name] = {"type": "error", "msg": str(e)}
        elif name == "fire_smoke":
            # Fire/smoke detection - classification
            img_rgb_norm = img_rgb / 255.0
            inputs = {models[name].input_name: img_rgb_norm[np.newaxis, ...]}
            try:
                outputs = models[name].sess.run(None, inputs)
                probs = outputs[0][0]
                result = {}
                for ci, v in enumerate(probs):
                    result["class_" + str(ci)] = float(v)
                result["max_class"] = int(np.argmax(probs))
                result["max_conf"] = float(np.max(probs))
                results[name] = result
                
                if np.max(probs) > 0.5:
                    max_cls = np.argmax(probs)
                    conf = float(probs[max_cls])
                    color = (0, 0, 255) if max_cls == 3 else (255, 255, 0)  # red for fire, yellow for smoke
                    txt = f"fire_smoke: class_{max_cls} {conf:.1%}"
                    cv2.putText(annotated, txt, (5, 30 + len(models) * 35 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                results[name] = {"type": "error", "msg": str(e)}
        elif name == "damage":
            # Damage detection - classification
            img_rgb_norm = img_rgb / 255.0
            inputs = {models[name].input_name: img_rgb_norm[np.newaxis, ...]}
            try:
                outputs = models[name].sess.run(None, inputs)
                probs = outputs[0][0]
                labels = {0: "no_damage", 1: "minor_damage", 
                          2: "severe_damage", 3: "destroyed"}
                max_dmg = int(np.argmax(probs))
                conf = float(probs[max_dmg])
                cls_name = labels.get(max_dmg, "unknown")
                results[name] = {"type": "damage", "class": cls_name, "conf": conf}
                
                color = (0, 255, 0) if max_dmg == 0 else (255, 0, 0)
                txt = f"damage: {cls_name} {conf:.1%}"
                cv2.putText(annotated, txt, (5, 30 + len(models) * 35 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                results[name] = {"type": "error", "msg": str(e)}
    
    return annotated, results


# --- HTTP Handler ---

class LiveHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.img_size = kwargs.pop('img_size', 224)
        self.shutter = kwargs.pop('shutter', 8000)
        self.models = kwargs.pop('models')
        super().__init__(*args, **kwargs)


class MJPEGHandler(LiveHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """
            <html><head><title>CANSAT - AI en Vivo</title>
            <style>body{margin:0;background:#000}
            img{width:100vw;height:auto}
            .info{color:#0f0;font-family:monospace;padding:10px}
            </style></head><body>
            <div class="info">CANSAT Expo • AI Camera IMX500 • Multi-model ONNX</div>
            <img src="/stream"></body></html>
            """
            self.wfile.write(html.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            frame_id = 0
            results_buf = {"val": None}
            results_lock = threading.Lock()
            
            while True:
                frame_id += 1
                annotated, results = self.capture_and_infer()
                with results_lock:
                    results_buf["val"] = results
                
                # Encode as JPEG
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                frame_bytes = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(buf)}\r\n\r\n".encode()
                ) + buf.tobytes() + b"\r\n"
                
                self.wfile.write(frame_bytes)
                
                # Append results to first frame's data
                with results_lock:
                    results_buf["val"] = results

        elif self.path == "/results":
            with results_lock:
                data = results_buf["val"] or {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def capture_and_infer(self):
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
        cam.start()
        time.sleep(0.5)  # wait for camera to stabilize
        cam.set_controls({"AeEnable": False, "AwbEnable": False})  # manual
        cam.set_controls({"AeEnable": True, "AwbEnable": True})  # re-enable for capture
        frame = cam.capture_array()
        cam.stop()
        
        annotated, results = annotate_frame(frame, self.models, self.img_size, self.shutter)
        return annotated, results

    def log_message(self, format, *args):
        pass  # Suppress logging


class StreamHandler(LiveHandler):
    results = {"val": None}
    lock = threading.Lock()
    
    def do_GET(self):
        import cv2
        from picamera2 import Picamera2
        
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """
            <html><head><title>CANSAT - AI en Vivo</title>
            <style>body{margin:0;background:#000}
            img{width:100vw;height:auto}
            .info{color:#0f0;font-family:monospace;padding:10px}
            </style></head><body>
            <div class="info">CANSAT Expo • AI Camera IMX500 • Multi-model ONNX</