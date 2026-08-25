"""
EmotionAI Web — Backend Flask
Raspberry Pi 4 + YOLOv8 Face + Classifier TFLite
Respaldo del 18/8/26
"""

from collections import deque
import cv2
import numpy as np
import time
import threading
import logging
from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO

from database import init_db

# Cola de 5 elementos para suavizar transiciones de forma rápida
emociones_recientes = deque(maxlen=5)

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_PATH_CPU  = "models/emociones.tflite"
INPUT_SIZE      = (224, 224)
CONFIDENCE_MIN  = 0.20
FRAME_SKIP      = 2

EMOTION_LABELS = ["enojado", "disgustado", "miedo", "feliz", "triste", "sorprendido", "neutral"]

state = {
    "emotion":       None,
    "confidence":    0.0,
    "fps":           0,
    "tpu":           "CPU TFLite",
    "face_count":    0,
    "current_frame": None,
}
state_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Cargar modelo TFLite ──────────────────────────────────────────────────────
def load_model():
    interp = tflite.Interpreter(model_path=MODEL_PATH_CPU)
    interp.allocate_tensors()
    return interp


# ── Preprocesamiento ──────────────────────────────────────────────────────────
def preprocess_face(face_img):
    """Preprocesamiento limpio sin distorsionar rasgos oscuros."""
    img = cv2.resize(face_img, INPUT_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3 canales idénticos en escala de grises
    img_3ch = cv2.merge([gray, gray, gray])
    
    # Normalización [-1, 1]
    img_normalized = (img_3ch.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(img_normalized, axis=0)

# ── Inferencia ────────────────────────────────────────────────────────────────
def run_inference(interpreter, face_tensor):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], face_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    probs = softmax(output) if output.max() > 1 else output
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])

    # ── Imprimir log en consola para ver los porcentajes reales ──
    logging.info(f"🔍 Predicción raw -> {EMOTION_LABELS[idx]} ({conf*100:.1f}%)")

    # ── Ajuste de Umbral ──
    # Bajamos de 0.45 a 0.32 para no "tapar" emociones como enojo o sorpresa
    if conf < 0.32:
        label = "neutral"
        conf = 0.50
    else:
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else "neutral"

    return label, conf

# ── Captura principal ─────────────────────────────────────────────────────────
def capture_loop():
    global state

    interpreter = load_model()
    face_detector = YOLO("yolov8n-face.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    fps_timer   = time.time()
    fps_val     = 0

    logging.info("🎥 Captura iniciada con YOLOv8 + TFLite")

    while True:
        try:
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_val     = frame_count
                frame_count = 0
                fps_timer   = now

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_streaming = frame.copy()

            # Detección de rostros con YOLO
            results = face_detector(frame, imgsz=320, conf=0.35, verbose=False)[0]

            faces = []
            if len(results.boxes) > 0:
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[0], x2), min(frame.shape[1], y2)
                    w, h = x2 - x1, y2 - y1
                    if w > 20 and h > 20:
                        faces.append((x1, y1, w, h))

            # Dibujar recuadros en la transmisión
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame_streaming, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

            ret_jpg, jpeg = cv2.imencode('.jpg', frame_streaming, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret_jpg:
                with state_lock:
                    state["current_frame"] = jpeg.tobytes()

            if frame_count % FRAME_SKIP != 0 or len(faces) == 0:
                if len(faces) == 0:
                    with state_lock:
                        state.update({"emotion": None, "confidence": 0.0, "fps": fps_val, "face_count": 0})
                continue

            # Tomar la cara principal y aplicar margen de 15%
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(0.25 * max(w, h))
            y1_crop = max(0, y - margin)
            y2_crop = min(frame.shape[0], y + h + margin)
            x1_crop = max(0, x - margin)
            x2_crop = min(frame.shape[1], x + w + margin)

            roi = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            tensor = preprocess_face(roi)
            label, conf = run_inference(interpreter, tensor)

            if conf < CONFIDENCE_MIN:
                label = None

            if label:
                emociones_recientes.append(label)

            emocion_estable = (
                max(set(emociones_recientes), key=emociones_recientes.count)
                if emociones_recientes else None
            )

            with state_lock:
                state.update({
                    "emotion":    emocion_estable,
                    "confidence": round(conf, 4),
                    "fps":        fps_val,
                    "face_count": len(faces),
                })

            time.sleep(0.01)

        except Exception as e:
            logging.error(f"Error en capture_loop: {e}")
            time.sleep(0.2)


# ── Endpoints Flask ───────────────────────────────────────────────────────────
def gen_frames():
    while True:
        with state_lock:
            frame = state["current_frame"]
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route("/")
def index():
    return render_template("emotionai_web.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/emotion")
def emotion():
    with state_lock:
        snap = dict(state)
    return jsonify({
        "emotion":    snap["emotion"] or "",
        "confidence": snap["confidence"],
        "fps":        snap["fps"],
        "tpu":        snap["tpu"],
        "face_count": snap["face_count"],
    })

if __name__ == "__main__":
    init_db()
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
