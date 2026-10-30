# Servidor web de telemetría LoRa en tiempo real para la expo
# Lee paquetes $LB135 desde COM7 (RX) y los muestra en navegador
# Usa Flask-like HTTP server (solo stdlib) + WebSocket opcional
import json
import threading
import time
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import re
import argparse

# ---- Serial reader thread ----
rx_data = {"val": None}
rx_history = []
rx_lock = threading.Lock()
lora_pattern = re.compile(r'\$(LB\d+)')

def serial_reader(port, baud):
    """Read LoRa packets from serial port and update shared data."""
    try:
        import serial
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[SERVER] Serial opened: {port} @ {baud}")
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', 'replace').strip()
                m = lora_pattern.search(line)
                if m:
                    pkt = parse_lb135(line)
                    if pkt:
                        with rx_lock:
                            rx_data["val"] = pkt
                            rx_history.append(pkt)
                        if len(rx_history) > 2000:
                            rx_history.clear()
            time.sleep(0.05)
        ser.close()
    except ImportError:
        print("[SERVER] pyserial not installed. pip install pyserial")
        sys.exit(1)
    except Exception as e:
        print(f"[SERVER] Serial error: {e}")

def parse_lb135(line):
    """Parse $LB135,x,x,... packet"""
    try:
        parts = line.split(',')
        if len(parts) < 24:
            return None
        pkt = {
            "raw": line,
            "header": parts[0],  # $LB135
            "version": int(parts[1]) if parts[1].isdigit() else 0,
            "seq": int(parts[2]) if parts[2].isdigit() else 0,
            "time_gps": parts[3],
            "lat_raw": float(parts[4]) if parts[4] else None,
            "lon_raw": float(parts[5]) if parts[5] else None,
            "pressure_hPa": float(parts[6]) if parts[6] else None,
            "temperature_C": float(parts[7]) if parts[7] else None,
            "humidity": float(parts[8]) if parts[8] else None,
            "acc_x": float(parts[9]) if parts[9] else None,
            "acc_y": float(parts[10]) if parts[10] else None,
            "acc_z": float(parts[11]) if parts[11] else None,
            "in_flight": int(parts[12]) if parts[12] else None,
            "vbat": float(parts[13]) if parts[13] else None,
            "gyro_x": float(parts[14]) if parts[14] else None,
            "gyro_y": float(parts[15]) if parts[15] else None,
            "gyro_z": float(parts[16]) if parts[16] else None,
            "heading": float(parts[21]) if len(parts) > 21 else None,
            "lat": float(parts[22]) if len(parts) > 22 else None,
            "lon": float(parts[23]) if len(parts) > 23 else None,
            "alt": float(parts[24]) if len(parts) > 24 else None,
            "rssi": None,
        }
        # Try to extract RSSI from comment line
        if line.endswith('*'):
            base = line.rsplit('*', 1)[0]
            if 'rssi=' in base:
                rssi_str = base.split('rssi=')[-1].split()[0]
                try:
                    pkt["rssi"] = float(rssi_str)
                except: pass
        return pkt
    except Exception:
        return None

# ---- HTTP Handler ----
class TelemetryHandler(SimpleHTTPRequestHandler):
    """Serves HTML dashboard + JSON telemetry API."""

    def log_message(self, format, *args):
        pass  # Suppress

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_html()
        elif self.path == '/telemetry':
            self.send_telemetry()
        elif self.path == '/history':
            seq_from = 0
            if '?' in self.path:
                try:
                    seq_from = int(self.path.split('?')[-1].split('=')[-1])
                except: seq_from = 0
            with rx_lock:
                history = [h for h in rx_history if h["seq"] > seq_from][-100:]
            self.send_json({"history": history, "count": len(history)})
        elif self.path == '/status':
            with rx_lock:
                last = rx_data["val"]
            status = {
                "port": port,
                "baud": baud,
                "has_data": last is not None,
                "seq": last["seq"] if last else -1,
                "rssi": last["rssi"] if last else None,
                "vbat": last["vbat"] if last else None,
            }
            self.send_json(status)
        elif self.path == '/gps-map':
            # Send last GPS position as GeoJSON
            with rx_lock:
                last = rx_data["val"]
            if last and last["lat"] and last["lon"]:
                geo = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [last["lon"], last["lat"]]},
                    "properties": {
                        "seq": last["seq"],
                        "alt": last.get("alt"),
                        "rssi": last.get("rssi"),
                        "vbat": last.get("vbat"),
                        "time": last.get("time_gps"),
                    }
                }
                self.send_json(geo)
            else:
                self.send_json({"error": "No GPS fix yet"})
        else:
            self.send_error(404)

    def send_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

# ---- HTML Dashboard ----
html_content = """\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CANSAT — Telemetría en Vivo</title>
<style>
  :root {
    --bg: #0a0e17; --card: #141a2a; --green: #00ff88; --cyan: #00ccff;
    --yellow: #ffcc00; --red: #ff4466; --orange: #ff8800; --text: #e0e8ff;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', 'Fira Code', monospace; }
  
  .header { padding:16px 24px; background:linear-gradient(135deg, #1a1f35, #0d1525); border-bottom:2px solid var(--green); }
  .header h1 { font-size:1.4em; color:var(--green); letter-spacing:1px; }
  .header .sub { font-size:.75em; color:#6a8fbf; margin-top:4px; }
  
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; padding:16px; max-width:1400px; margin:auto; }
  
  .card { background:var(--card); border-radius:10px; padding:16px; border:1px solid #1e2a45; box-shadow:0 4px 12px #00000040; }
  .card h2 { font-size:.85em; color:var(--cyan); text-transform:uppercase; letter-spacing:2px; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #1e2a4566; }
  
  .telemetry-row { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:.85em; }
  .telemetry-row .label { color:#6a8fbf; }
  .telemetry-row .value { color:var(--green); font-weight:bold; }
  .telemetry-row .value.warn { color:var(--yellow); }
  .telemetry-row .value.crit { color:var(--red); }
  
  .status-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    background:var(--green); margin-right:6px; animation:blink 1s infinite;
  }
  .status-dot.offline { background:var(--red); animation:none; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
  
  .packet-list { max-height:300px; overflow-y:auto; font-size:.75em; }
  .packet-list::-webkit-scrollbar { width:4px; }
  .packet-list::-webkit-scrollbar-thumb { background:#334; border-radius:2px; }
  .pkt-row { padding:3px 8px; border-bottom:1px solid #1a2240; display:flex; justify-content:space-between; }
  .pkt-row:hover { background:#1a2240; }
  .pkt-seq { color:var(--cyan); }
  .pkt-vbat { color:var(--yellow); }
  .pkt-rssi { color:var(--green); }
  
  .big-value { font-size:2.5em; text-align:center; padding:10px 0; }
  .big-unit { font-size:.4em; color:#6a8fbf; }
  
  .rssi-bar { height:8px; background:#1e2a45; border-radius:4px; margin-top:6px; overflow:hidden; }
  .rssi-fill { height:100%; background:var(--green); transition:width .5s, background .5s; }
  
  @media (max-width:768px) {
    .grid { grid-template-columns:1fr; padding:8px; }
    .big-value { font-size:1.8em; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span class="status-dot" id="dot"></span>CANSAT — Telemetría en Vivo</h1>
  <div class="sub">RX LoRa • SF9 BW125 • <span id="conn-status">Conectando...</span></div>
</div>

<div class="grid">
  <!-- GPS Card -->
  <div class="card">
    <h2>📍 Posición GPS</h2>
    <div class="telemetry-row"><span class="label">Latitud</span><span class="value" id="lat">--</span></div>
    <div class="telemetry-row"><span class="label">Longitud</span><span class="value" id="lon">--</span></div>
    <div class="telemetry-row"><span class="label">Altitud</span><span class="value" id="alt">--</span></div>
    <div class="telemetry-row"><span class="label">Satélites</span><span class="value" id="sat">--</span></div>
    <div class="telemetry-row"><span class="label">UTC</span><span class="value" id="utc">--</span></div>
  </div>

  <!-- Sensors Card -->
  <div class="card">
    <h2>🌡️ Sensores</h2>
    <div class="telemetry-row"><span class="label">Temperatura</span><span class="value" id="temp">--</span></div>
    <div class="telemetry-row"><span class="label">Presión</span><span class="value" id="pres">--</span></div>
    <div class="telemetry-row"><span class="label">Humedad</span><span class="value" id="hum">--</span></div>
    <div class="telemetry-row"><span class="label">VBAT</span><span class="value" id="vbat">--</span></div>
    <div class="telemetry-row"><span class="label">Aceleración</span><span class="value" id="acc">--</span></div>
  </div>

  <!-- LoRa Card -->
  <div class="card">
    <h2>📡 LoRa</h2>
    <div class="telemetry-row"><span class="label">RSSI</span><span class="value" id="rssi">--</span></div>
    <div class="rssi-bar"><div class="rssi-fill" id="rssi-bar" style="width:0%"></div></div>
    <div class="telemetry-row"><span class="label">Último seq</span><span class="value" id="last-seq">--</span></div>
    <div class="telemetry-row"><span class="label">Total recibidos</span><span class="value" id="total">--</span></div>
  </div>

  <!-- IMU Card -->
  <div class="card">
    <h2>🧭 IMU (MPU6050)</h2>
    <div class="telemetry-row"><span class="label">Gyro X</span><span class="value" id="gyx">--</span></div>
    <div class="telemetry-row"><span class="label">Gyro Y</span><span class="value" id="gyy">--</span></div>
    <div class="telemetry-row"><span class="label">Gyro Z</span><span class="value" id="gyz">--</span></div>
    <div class="telemetry-row"><span class="label">En vuelo</span><span class="value" id="inv">--</span></div>
  </div>
</div>

<!-- Full packet history -->
<div style="padding:0 16px; max-width:1400px; margin:auto;">
  <div class="card">
    <h2>📋 Historial de Paquetes</h2>
    <div class="packet-list" id="pkt-list"></div>
  </div>
</div>

<script>
  const dot = document.getElementById('dot');
  const connStatus = document.getElementById('conn-status');
  let lastSeq = -1;

  async function updateStatus() {
    try {
      const r = await fetch('/status?t=' + Date.now());
      const s = await r.json();
      if (s.has_data) {
        dot.classList.remove('offline');
        connStatus.textContent = `✅ ${s.port} @ ${s.baud} • ${s.seq}`;
      } else {
        dot.classList.add('offline');
        connStatus.textContent = '⏳ Esperando RX...';
      }
    } catch {
      dot.classList.add('offline');
      connStatus.textContent = '❌ Desconectado';
    }
  }

  async function updateTelemetry() {
    try {
      const r = await fetch('/telemetry?t=' + Date.now());
      const t = await r.json();
      const d = t.data || t.val;
      if (!d) return;
      
      document.getElementById('lat').textContent = d.lat != null ? d.lat.toFixed(6) : '--';
      document.getElementById('lon').textContent = d.lon != null ? d.lon.toFixed(6) : '--';
      document.getElementById('alt').textContent = d.alt != null ? d.alt.toFixed(1) + 'm' : '--';
      document.getElementById('utc').textContent = d.time_gps || '--';
      document.getElementById('temp').textContent = d.temperature_C != null ? d.temperature_C.toFixed(1) + '°C' : '--';
      document.getElementById('pres').textContent = d.pressure_hPa != null ? d.pressure_hPa.toFixed(1) + ' hPa' : '--';
      document.getElementById('hum').textContent = d.humidity != null ? d.humidity.toFixed(1) + '%' : '--';
      
      const vbatEl = document.getElementById('vbat');
      vbatEl.textContent = d.vbat != null ? d.vbat.toFixed(2) + 'V' : '--';
      vbatEl.className = 'value' + (d.vbat != null && d.vbat < 3.6 ? ' crit' : '');

      const rssiEl = document.getElementById('rssi');
      rssiEl.textContent = d.rssi != null ? d.rssi.toFixed(1) + ' dBm' : '--';
      if (d.rssi != null) {
        const bar = document.getElementById('rssi-bar');
        const pct = Math.max(0, Math.min(100, -d.rssi + 60)); // -61 -> 1%, -30 -> 70%
        bar.style.width = pct + '%';
        bar.style.background = d.rssi > -45 ? 'var(--green)' : (d.rssi > -55 ? 'var(--yellow)' : 'var(--red)');
      }

      document.getElementById('last-seq').textContent = d.seq;
      document.getElementById('acc').textContent = `(${d.acc_x?.toFixed(2)??0}, ${d.acc_y?.toFixed(2)??0}, ${d.acc_z?.toFixed(2)??0})`;
      document.getElementById('gyx').textContent = d.gyro_x != null ? d.gyro_x.toFixed(3) : '--';
      document.getElementById('gyy').textContent = d.gyro_y != null ? d.gyro_y.toFixed(3) : '--';
      document.getElementById('gyz').textContent = d.gyro_z != null ? d.gyro_z.toFixed(3) : '--';
      document.getElementById('inv').textContent = d.in_flight ? '✅ Sí' : '⏳ No';
      document.getElementById('sat').textContent = '30'; // placeholder from last fix
    } catch(e) {
      console.log('Telemetry fetch error:', e);
    }
  }

  async function updateHistory() {
    try {
      const r = await fetch('/history=' + (lastSeq || 0) + '?t=' + Date.now());
      const h = await r.json();
      const list = document.getElementById('pkt-list');
      for (const pkt of h.history) {
        const row = document.createElement('div');
        row.className = 'pkt-row';
        row.innerHTML = `
          <span class="pkt-seq">#${pkt.seq}</span>
          <span>VBAT: <span class="pkt-vbat">${pkt.vbat?.toFixed(2)??0}V</span></span>
          <span>RSSI: <span class="pkt-rssi">${pkt.rssi != null ? pkt.rssi.toFixed(0) : '--'}</span> dBm</span>
          <span>${pkt.time_gps || ''}</span>
        `;
        list.insertBefore(row, list.firstChild);
        lastSeq = pkt.seq;
      }
      while (list.children.length > 50) list.removeChild(list.lastChild);
    } catch(e) { console.log('History error:', e); }
  }

  // Poll
  setInterval(updateStatus, 2000);
  setInterval(updateTelemetry, 1000);
  setInterval(updateHistory, 2000);
  updateStatus();
  updateTelemetry();
  updateHistory();
</script>
</body>
</html>
"""

# ---- Main ----
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8765, help='HTTP port')
    p.add_argument('--com', default='COM7', help='RX COM port')
    p.add_argument('--baud', type=int, default=115200)
    args = p.parse_args()

    port, baud = args.com, args.baud
    t = threading.Thread(target=serial_reader, args=(port, baud), daemon=True)
    t.start()

    httpd = HTTPServer(('0.0.0.0', args.port), TelemetryHandler)
    print('\n============================================================')
    ascii_banner = f'  [SERVER] CANSAT Telemetry Dashboard: http://localhost:{args.port}'
    print(ascii_banner)
    print(f'  [SERIAL] Port: {port} @ {baud}')
    print('============================================================\n')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n\nShutting down...')
        httpd.shutdown()
