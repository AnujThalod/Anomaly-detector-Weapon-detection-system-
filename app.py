from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import base64
import requests
import numpy as np
import subprocess
import threading
import time
import urllib.parse
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', template_folder='templates')

model = YOLO("best_anomaly_model.pt")
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')
@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[1].lower()

    if ext in {'jpg', 'jpeg', 'png'}:
        return process_image(filepath, filename)
    elif ext in {'mp4', 'avi', 'mov', 'mkv', 'webm'}:
        return process_video(filepath, filename)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


def process_image(filepath, filename):
    try:
        results = model(filepath)
        img = cv2.imread(filepath)

        if img is None:
            return jsonify({'error': 'Could not read image file'}), 400

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names.get(cls, str(cls))}: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        processed_path = os.path.join(PROCESSED_FOLDER, filename)
        cv2.imwrite(processed_path, img)
        return jsonify({'type': 'image', 'path': url_for('static', filename=f'processed/{filename}')})

    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500


def process_video(filepath, filename):
    try:
        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file. Format may be unsupported.'}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Fix: agar width/height 0 aaye toh error do
        if width == 0 or height == 0:
            cap.release()
            return jsonify({'error': 'Could not read video dimensions. Try a different format.'}), 400

        # Fix: agar fps 0 ya invalid aaye toh default use karo
        if fps <= 0 or fps > 120:
            fps = 25.0

        # Pehle raw opencv output banao (temp file)
        base_name = filename.rsplit('.', 1)[0]
        temp_name = base_name + '_temp.mp4'
        processed_name = base_name + '_processed.mp4'
        temp_path = os.path.join(PROCESSED_FOLDER, temp_name)
        processed_path = os.path.join(PROCESSED_FOLDER, processed_name)

        # mp4v codec se temp file likhte hain
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return jsonify({'error': 'Could not create output video writer.'}), 500

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                results = model(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = f"{model.names.get(cls, str(cls))}: {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2)
            except Exception as frame_err:
                print(f"Frame {frame_count} error: {frame_err}")
                # Frame skip karo, baaki video chalti rahe

            out.write(frame)

        cap.release()
        out.release()

        if frame_count == 0:
            return jsonify({'error': 'Video has no readable frames.'}), 400

        # Fix: ffmpeg se re-encode karo H.264 mein taaki browser pe chale
        ffmpeg_ok = convert_to_h264(temp_path, processed_path)

        if ffmpeg_ok:
            if os.path.exists(temp_path):
                os.remove(temp_path)  # temp file delete karo
        else:
            # ffmpeg nahi hai toh temp file ko hi use karo
            if os.path.exists(processed_path):
                 os.remove(processed_path)
            os.rename(temp_path, processed_path)

        return jsonify({
            'type': 'video',
            'path': url_for('static', filename=f'processed/{processed_name}')
        })

    except Exception as e:
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500


def convert_to_h264(input_path, output_path):
    """
    ffmpeg se H.264 mein convert karo taaki browser mein play ho.
    Returns True agar successful, False agar ffmpeg available nahi.
    """
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-strict', 'experimental',
                '-movflags', '+faststart',  # Web streaming ke liye zaroori
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 min timeout
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"ffmpeg conversion failed: {e}")
        return False

## ══════════════════════════════════════════════════
## IP CAMERA STREAM MANAGER (Background Thread)
## ══════════════════════════════════════════════════

class IPCameraStream:
    """
    Background thread se IP camera ka stream continuously read karta hai.
    Two modes:
      1. STREAM mode: cv2.VideoCapture for MJPEG/RTSP continuous streams
      2. SNAPSHOT mode: requests-based polling of /shot.jpg endpoint (more reliable fallback)
    """
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.latest_frame = None
        self.last_access_time = time.time()
        self.is_running = False
        self.error_message = None
        self.thread = None
        self.lock = threading.Lock()
        self.connected = False
        self.mode = None  # 'stream' or 'snapshot'

    def start(self):
        self.is_running = True
        self.error_message = None
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _normalize_url(self, url):
        """Add http:// if no protocol given."""
        if not url.startswith(('http://', 'https://', 'rtsp://')):
            url = 'http://' + url
        return url.rstrip('/')

    def _check_reachable(self, base_url):
        """Quick HTTP check to see if the IP camera is reachable at all."""
        try:
            r = requests.get(base_url, timeout=3)
            return True
        except:
            return False

    def _try_snapshot(self, snapshot_url):
        """Try fetching a single JPEG snapshot via requests. Returns frame or None."""
        try:
            r = requests.get(snapshot_url, timeout=5)
            if r.status_code == 200 and len(r.content) > 1000:
                nparr = np.frombuffer(r.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except:
            pass
        return None

    def _try_stream(self, stream_url):
        """Try opening a cv2.VideoCapture stream. Returns (cap, frame) or (None, None)."""
        try:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Give it up to 5 seconds to connect
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, frame
                cap.release()
        except Exception as e:
            print(f"[IPCam] Stream open error for {stream_url}: {e}")
        return None, None

    def _update(self):
        """Background thread: try multiple connection methods."""
        base_url = self._normalize_url(self.url)
        parsed = urllib.parse.urlparse(base_url)
        has_path = parsed.path and parsed.path != '/'

        print(f"[IPCam] Starting connection to: {base_url}")

        # Step 1: Quick reachability check
        if parsed.scheme in ('http', 'https'):
            check_url = f"{parsed.scheme}://{parsed.netloc}"
            print(f"[IPCam] Checking reachability: {check_url}")
            if not self._check_reachable(check_url):
                self.error_message = (
                    "Cannot reach the IP camera at all!\n"
                    "Please check:\n"
                    "• Is the IP camera app running on your phone?\n"
                    "• Are phone and computer on the SAME WiFi?\n"
                    f"• Try opening {check_url} in your browser\n"
                    "• Check if Windows Firewall is blocking the connection"
                )
                self.is_running = False
                return

        print("[IPCam] Camera is reachable! Trying to connect to stream...")

        # Step 2: Build candidate list
        if has_path:
            # User specified a specific path, use it directly
            snapshot_candidates = []
            stream_candidates = [base_url]
            # Also check if it's a snapshot URL
            if base_url.endswith(('.jpg', '.jpeg', '.png')):
                snapshot_candidates = [base_url]
                stream_candidates = []
        else:
            # Root URL — try common endpoints
            snapshot_candidates = [
                base_url + '/shot.jpg',       # Android IP Webcam
            ]
            stream_candidates = [
                base_url + '/video',           # Android IP Webcam MJPEG
                base_url + '/mjpegfeed',       # Some apps
                base_url + '/stream',          # Generic
            ]

        # Step 3: Try snapshot mode first (fastest and most reliable)
        for snap_url in snapshot_candidates:
            print(f"[IPCam] Trying snapshot: {snap_url}")
            frame = self._try_snapshot(snap_url)
            if frame is not None:
                print(f"[IPCam] ✓ Snapshot mode connected: {snap_url}")
                self.mode = 'snapshot'
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                # Snapshot loop
                self._run_snapshot_loop(snap_url)
                return

        # Step 4: Try video stream mode
        for stream_url in stream_candidates:
            print(f"[IPCam] Trying stream: {stream_url}")
            cap, frame = self._try_stream(stream_url)
            if cap is not None:
                print(f"[IPCam] ✓ Stream mode connected: {stream_url}")
                self.cap = cap
                self.mode = 'stream'
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                # Stream loop
                self._run_stream_loop()
                return

        # Nothing worked
        self.error_message = (
            "Camera is reachable but cannot get video stream.\n"
            "Please try entering one of these URLs directly:\n"
            f"• {base_url}/video\n"
            f"• {base_url}/shot.jpg\n"
            "Or check what URL your IP camera app shows."
        )
        self.is_running = False

    def _run_snapshot_loop(self, snapshot_url):
        """Keep fetching snapshots from /shot.jpg endpoint."""
        retry_count = 0
        while self.is_running:
            if time.time() - self.last_access_time > 15.0:
                print("[IPCam] No activity for 15s, shutting down.")
                break

            try:
                r = requests.get(snapshot_url, timeout=3)
                if r.status_code == 200 and len(r.content) > 500:
                    nparr = np.frombuffer(r.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.error_message = None
                        retry_count = 0
                    else:
                        retry_count += 1
                else:
                    retry_count += 1
            except:
                retry_count += 1

            if retry_count > 20:
                with self.lock:
                    self.error_message = "Lost connection to IP camera."
                break

            time.sleep(0.05)  # ~20 FPS max

        self.is_running = False
        self.connected = False
        print("[IPCam] Snapshot loop stopped.")

    def _run_stream_loop(self):
        """Keep reading frames from cv2.VideoCapture stream."""
        retry_count = 0
        while self.is_running:
            if time.time() - self.last_access_time > 15.0:
                print("[IPCam] No activity for 15s, shutting down.")
                break

            ret, frame = self.cap.read()
            if not ret:
                retry_count += 1
                if retry_count > 30:
                    with self.lock:
                        self.error_message = "Lost connection to IP camera."
                    break
                time.sleep(0.1)
                continue

            retry_count = 0
            with self.lock:
                self.latest_frame = frame
                self.error_message = None

        if self.cap:
            self.cap.release()
        self.is_running = False
        self.connected = False
        print("[IPCam] Stream loop stopped.")

    def get_frame(self):
        """Frontend calls this — returns latest frame instantly."""
        self.last_access_time = time.time()
        with self.lock:
            return self.latest_frame, self.error_message

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        self.connected = False


# Global camera stream manager
active_streams = {}
streams_lock = threading.Lock()

# Detection cache — so YOLO doesn't run on every single frame
# Key: cam_url, Value: {'detections': [...], 'time': timestamp}
detection_cache = {}
detection_cache_lock = threading.Lock()
DETECTION_INTERVAL = 0.8  # Run YOLO at most once every 0.8 seconds
CONFIDENCE_THRESHOLD = 0.45  # Higher = fewer false positives
MODEL_IMG_SIZE = 416  # Smaller = faster inference


def get_or_create_stream(url):
    """Get existing stream or create a new one for this URL."""
    with streams_lock:
        # Clean up dead streams
        dead_keys = [k for k, v in active_streams.items() if not v.is_running]
        for k in dead_keys:
            del active_streams[k]

        # Check if stream already exists and is running
        if url in active_streams and active_streams[url].is_running:
            return active_streams[url]

        # Create new stream
        stream = IPCameraStream(url)
        stream.start()
        active_streams[url] = stream
        return stream


def run_detection(frame):
    """Run YOLO detection on a frame with optimized settings."""
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=MODEL_IMG_SIZE)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'label': label
            })
    return detections


## ══════════════════════════════════════════════════
## ROUTES
## ══════════════════════════════════════════════════

@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """
    Webcam se aaya ek frame detect karo.
    Frontend har 500ms pe ek JPEG frame bhejta hai.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No frame received'}), 400

        file = request.files['file']
        img_bytes = file.read()

        # Bytes se OpenCV image banao
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'detections': []})

        # YOLO se detect karo
        detections = run_detection(frame)

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_ip_frame', methods=['POST'])
def detect_ip_frame():
    """
    IP Camera se latest frame lo aur return karo.
    YOLO detection sirf har ~1 second mein chalta hai (cached results).
    Frame HAMESHA turant return hota hai = smooth video.
    """
    try:
        data = request.get_json()
        cam_url = data.get('url', '').strip()

        if not cam_url:
            return jsonify({'error': 'No camera URL provided'}), 400

        # Get or create background stream
        stream = get_or_create_stream(cam_url)

        # Give new streams time to connect (max 8 seconds for first connection)
        if not stream.connected and stream.is_running:
            for _ in range(80):
                time.sleep(0.1)
                if stream.connected or not stream.is_running:
                    break

        frame, error = stream.get_frame()

        if error:
            return jsonify({'error': error}), 400

        if frame is None:
            if stream.is_running:
                return jsonify({'error': 'Still connecting to camera... please wait'}), 202
            else:
                return jsonify({'error': 'Could not connect to IP camera. Check URL and network.'}), 400

        # ── SMART DETECTION: only run YOLO every DETECTION_INTERVAL seconds ──
        now = time.time()
        detections = []
        should_detect = False

        with detection_cache_lock:
            cached = detection_cache.get(cam_url)
            if cached is None or (now - cached['time']) >= DETECTION_INTERVAL:
                should_detect = True
            else:
                detections = cached['detections']

        if should_detect:
            detections = run_detection(frame)
            with detection_cache_lock:
                detection_cache[cam_url] = {
                    'detections': detections,
                    'time': now
                }

        # Frame ko base64 mein encode karo frontend ke liye
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'frame': frame_b64,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop_ip_stream', methods=['POST'])
def stop_ip_stream():
    """Stop a running IP camera stream."""
    try:
        data = request.get_json()
        cam_url = data.get('url', '').strip()
        with streams_lock:
            if cam_url in active_streams:
                active_streams[cam_url].stop()
                del active_streams[cam_url]
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

