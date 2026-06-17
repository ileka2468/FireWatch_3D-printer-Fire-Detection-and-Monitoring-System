import threading
import logging
import cv2
from flask import Flask, render_template, Response, jsonify
import os

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global references
_stream_manager = None
_alert_manager = None

def generate_frames(stream_name: str):
    """Generator for MJPEG streaming."""
    global _stream_manager
    while True:
        if not _stream_manager:
            break
            
        frame = _stream_manager.get_frame(stream_name)
        if frame is None:
            import time
            time.sleep(0.1)
            continue
            
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main dashboard HTML."""
    return render_template('index.html')

@app.route('/api/state')
def api_state():
    """Return JSON state for the frontend dashboard."""
    global _stream_manager, _alert_manager
    if not _stream_manager or not _alert_manager:
        return jsonify({"error": "Managers not initialized"}), 500
        
    streams = []
    names = _stream_manager.get_stream_names()
    statuses = _stream_manager.get_all_status()
    
    for name in names:
        streams.append({
            "name": name,
            "connected": statuses.get(name, False)
        })
        
    logs = []
    for entry in _alert_manager.alert_history[-10:]:
        logs.append(f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.stream_name} - {entry.status}: {entry.description}")
        
    return jsonify({
        "streams": streams,
        "alarm_active": _alert_manager.alarm_active,
        "logs": logs
    })

@app.route('/video_feed/<stream_name>')
def video_feed(stream_name):
    """MJPEG stream endpoint."""
    # Decode URL-safe stream name if needed, but Flask handles basic unquoting
    return Response(generate_frames(stream_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask(port: int):
    # Disable flask output to avoid cluttering console
    import logging as flask_logging
    log = flask_logging.getLogger('werkzeug')
    log.setLevel(flask_logging.ERROR)
    
    # Check if we are running in a PyInstaller bundle
    import sys
    from core.config_manager import get_resource_path
    
    # We must configure Flask to look in the correct templates folder
    template_dir = os.path.abspath(str(get_resource_path("web/templates")))
    static_dir = os.path.abspath(str(get_resource_path("web/static")))
    
    app.template_folder = template_dir
    app.static_folder = static_dir
    
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)

def start_server(stream_manager, alert_manager, port: int = 5050) -> threading.Thread:
    """Start the Flask server in a background thread."""
    global _stream_manager, _alert_manager
    _stream_manager = stream_manager
    _alert_manager = alert_manager
    
    thread = threading.Thread(target=run_flask, args=(port,), daemon=True, name="web-server")
    thread.start()
    logger.info(f"Web dashboard started on port {port}")
    return thread
