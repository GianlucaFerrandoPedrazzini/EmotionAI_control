"""
EmotionAI Web — Backend Flask
Raspberry Pi 4 + RB Cam + MediaPipe Face Mesh (Basado en Landmarks)
Detencion
"""

import cv2
import numpy as np
import time
import threading
import logging
import math
from collections import deque
from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Crear el detector usando la nueva Tasks API
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

from database import init_db, save_detection

# Cola suavizada para estabilizar la emoción
emociones_recientes = deque(maxlen=5)

# ── Estado global ─────────────────────────────────────────────────────────────
state = {
    "emotion":       None,
    "confidence":    0.0,
    "fps":           0,
    "tpu":           "MediaPipe Landmarks",
    "face_count":    0,
    "current_frame": None,
}
state_lock = threading.Lock()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Función para calcular distancia euclidiana ────────────────────────────────
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


# ── Clasificador Geométrico de Emociones ──────────────────────────────────────
def analyze_landmarks(landmarks):
    """
    Analiza la geometría de los 478 puntos del rostro.
    Devuelve (emoción, confianza)
    """
    # Puntos Clave de MediaPipe Face Mesh:
    # 61, 291 : Comisuras de la boca (extremos)
    # 13, 14  : Labio superior e inferior (centro)
    # 33, 263 : Esquinas exteriores de los ojos (para normalizar escala)
    # 70, 300 : Cejas (parte superior)
    # 159, 145: Ojo izquierdo (arriba/abajo)

    # 1. Normalizador de escala (Ancho de la cara basado en la distancia entre ojos)
    eye_dist = distance(landmarks[33], landmarks[263])
    if eye_dist == 0:
        return "neutral", 0.50

    # 2. Métricas de la boca normalizadas
    mouth_width = distance(landmarks[61], landmarks[291]) / eye_dist
    mouth_height = distance(landmarks[13], landmarks[14]) / eye_dist

    # 3. Métricas de cejas normalizadas (Distancia entre ceja y ojo)
    eyebrow_left = distance(landmarks[70], landmarks[159]) / eye_dist
    eyebrow_right = distance(landmarks[300], landmarks[386]) / eye_dist
    eyebrow_avg = (eyebrow_left + eyebrow_right) / 2.0

    # ── Lógica de Reglas de Expresión ──

    # A. Sonrisa (Feliz): La boca se ensancha horizontalmente
    if mouth_width > 1.15:
        conf = min(0.99, round((mouth_width - 1.0) * 2.5, 2))
        return "feliz", conf

    # B. Sorpresa: Apertura vertical grande de boca + cejas elevadas
    if mouth_height > 0.40 and eyebrow_avg > 0.35:
        conf = min(0.99, round(mouth_height * 2.0, 2))
        return "sorprendido", conf

    # C. Enojo / Tristeza: Cejas fruncidas/bajas + boca estrecha
    if eyebrow_avg < 0.22:
        if mouth_height < 0.10:
            return "enojado", 0.85
        else:
            return "triste", 0.75

    # D. Estado por defecto: Neutral
    return "neutral", 0.80


# ── Hilo principal de captura (MediaPipe Tasks API) ───────────────────────────
def capture_loop():
    global state

    # Cargar el modelo .task desde la raíz del proyecto
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Iniciar la cámara con OpenCV
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    fps_timer = time.time()
    fps_val = 0
    FRAME_SKIP = 2  # Procesar inferencia cada 2 frames para optimizar CPU

    logging.info("🎥 Captura iniciada con MediaPipe Tasks API")

    while True:
        try:
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_val = frame_count
                frame_count = 0
                fps_timer = now

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_streaming = frame.copy()
            detected_emotion = None
            detected_conf = 0.0
            face_count = 0

            # Procesar detección de puntos clave en frames alternos
            if frame_count % FRAME_SKIP == 0:
                # Reducir imagen solo para procesar rápida la inferencia
                small_frame = cv2.resize(frame, (320, 240))
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a objeto mp.Image de MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
                
                # Inferencia
                detection_result = landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    face_count = len(detection_result.face_landmarks)
                    landmarks = detection_result.face_landmarks[0]
                    
                    # Analizar geometría de la cara
                    detected_emotion, detected_conf = analyze_landmarks(landmarks)

                    # Dibujar puntos clave en la imagen original de streaming
                    h, w, _ = frame_streaming.shape
                    puntos_clave_idx = [61, 291, 13, 14, 70, 300]
                    for idx in puntos_clave_idx:
                        pt = landmarks[idx]
                        cx, cy = int(pt.x * w), int(pt.y * h)
                        cv2.circle(frame_streaming, (cx, cy), 4, (0, 255, 0), -1)

            # Codificar siempre el frame JPEG para un video fluido en /video_feed
            ret_jpg, jpeg = cv2.imencode('.jpg', frame_streaming, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret_jpg:
                with state_lock:
                    state["current_frame"] = jpeg.tobytes()

            # Estabilizar y suavizar la emoción detectada
            if detected_emotion:
                emociones_recientes.append(detected_emotion)

            emocion_estable = (
                max(set(emociones_recientes), key=emociones_recientes.count)
                if emociones_recientes else None
            )

            # Actualizar estado global accesible por Flask
            with state_lock:
                state.update({
                    "emotion":       emocion_estable,
                    "confidence":    detected_conf,
                    "fps":           fps_val,
                    "tpu":           "MediaPipe Tasks",
                    "face_count":    face_count,
                })

            time.sleep(0.01)

        except Exception as e:
            logging.error(f"Error en capture_loop: {e}")
            time.sleep(0.2)


# ── Generador Streaming ───────────────────────────────────────────────────────
def gen_frames():
    while True:
        with state_lock:
            frame = state["current_frame"]

        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)


# ── Endpoints Flask ───────────────────────────────────────────────────────────
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
    logging.info("🚀 Server en http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
