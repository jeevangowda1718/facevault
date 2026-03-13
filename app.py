import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import threading
import time

# Lazy-load DeepFace to avoid slow startup
_deepface = None

def get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def get_face_embedding(frame_rgb):
    """Get face embedding using DeepFace Facenet model."""
    try:
        DeepFace = get_deepface()
        result = DeepFace.represent(
            img_path=frame_rgb,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        if result:
            return np.array(result[0]['embedding']), result[0]['facial_area']
        return None, None
    except Exception:
        return None, None


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / denom if denom else 0

app = Flask(__name__)

ENCODINGS_FILE = os.path.join('data', 'encodings.pkl')
os.makedirs('data', exist_ok=True)

# Haar cascade for fast face box drawing
_face_cascade = None

def get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _face_cascade


def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {'encodings': [], 'names': []}


def save_encodings(data):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)


def process_frame_for_recognition(frame):
    """Detect faces with Haar, match embeddings with DeepFace."""
    data = load_encodings()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        name = "Unknown"
        if data['encodings']:
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            embedding, _ = get_face_embedding(face_roi)
            if embedding is not None:
                best_sim, best_name = 0, "Unknown"
                for enc, n in zip(data['encodings'], data['names']):
                    sim = cosine_similarity(embedding, enc)
                    if sim > best_sim:
                        best_sim, best_name = sim, n
                if best_sim > 0.70:
                    name = best_name

        color = (0, 255, 180) if name != "Unknown" else (0, 100, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 32), (x + w, y), color, cv2.FILLED)
        cv2.putText(frame, name, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1)

    return frame


class VideoCamera:
    def __init__(self, mode='detect'):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.mode = mode
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.03)

    def get_frame(self):
        if self.frame is None:
            return None
        frame = self.frame.copy()
        if self.mode == 'detect':
            frame = process_frame_for_recognition(frame)
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes() if ret else None

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


detect_camera = None
register_camera = None


def gen_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/video_feed/register')
def video_feed_register():
    global register_camera
    if register_camera is None or not register_camera.running:
        register_camera = VideoCamera(mode='register')
    return Response(gen_frames(register_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/detect')
def video_feed_detect():
    global detect_camera
    if detect_camera is None or not detect_camera.running:
        detect_camera = VideoCamera(mode='detect')
    return Response(gen_frames(detect_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera/<mode>', methods=['POST'])
def stop_camera(mode):
    global detect_camera, register_camera
    if mode == 'detect' and detect_camera:
        detect_camera.release()
        detect_camera = None
    elif mode == 'register' and register_camera:
        register_camera.release()
        register_camera = None
    return jsonify({'status': 'stopped'})


@app.route('/save_face', methods=['POST'])
def save_face():
    global register_camera
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'}), 400
    if register_camera is None or register_camera.frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400

    encodings_collected = []
    attempts = 0

    while len(encodings_collected) < 12 and attempts < 50:
        frame = register_camera.frame
        if frame is not None:
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding, _ = get_face_embedding(face_rgb)
            if embedding is not None:
                encodings_collected.append(embedding)
        time.sleep(0.15)
        attempts += 1

    if not encodings_collected:
        return jsonify({
            'success': False,
            'message': 'No face detected. Ensure good lighting and face the camera directly.'
        }), 400

    avg_encoding = np.mean(encodings_collected, axis=0)
    data = load_encodings()

    if name in data['names']:
        idx = data['names'].index(name)
        existing = [e for e, n in zip(data['encodings'], data['names']) if n == name]
        existing.append(avg_encoding)
        data['encodings'][idx] = np.mean(existing, axis=0)
        message = f"Updated {name} with {len(encodings_collected)} frames"
    else:
        data['encodings'].append(avg_encoding)
        data['names'].append(name)
        message = f"Registered {name} successfully! ({len(encodings_collected)} frames captured)"

    save_encodings(data)
    return jsonify({'success': True, 'message': message, 'frames': len(encodings_collected)})


@app.route('/get_registered')
def get_registered():
    data = load_encodings()
    names = list(set(data['names']))
    return jsonify({'names': names, 'count': len(names)})


@app.route('/delete_face', methods=['POST'])
def delete_face():
    name = request.json.get('name', '').strip()
    data = load_encodings()
    indices = [i for i, n in enumerate(data['names']) if n != name]
    data['encodings'] = [data['encodings'][i] for i in indices]
    data['names'] = [data['names'][i] for i in indices]
    save_encodings(data)
    return jsonify({'success': True, 'message': f'Removed {name}'})


if __name__ == '__main__':
    print("=" * 50)
    print("  FaceVault — Starting...")
    print("  NOTE: First run downloads Facenet model (~90MB)")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)