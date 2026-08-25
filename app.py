from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import threading
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

app = Flask(__name__)

# ---------------------------------------------------------
# 1. Variables Globales y Configuración
# ---------------------------------------------------------
MODEL_TPU_PATH = "models/emociones_edgetpu.tflite"
MODEL_CPU_PATH = "models/emociones.tflite"
LIBEDGETPU_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
CLASSES = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

latest_emotion = "Cargando..."
current_frame = None
lock = threading.Lock()

# ---------------------------------------------------------
# 2. Carga del Modelo TFLite
# ---------------------------------------------------------
def create_interpreter(use_tpu=True):
    if use_tpu:
        try:
            interp = tflite.Interpreter(
                model_path=MODEL_TPU_PATH,
                experimental_delegates=[
                    tflite.load_delegate(LIBEDGETPU_PATH, {'device': 'usb:0'})
                ]
            )
            interp.allocate_tensors()
            print(">>> Modelo cargado exitosamente en TPU Coral.")
            return interp, True
        except Exception as e:
            print(f">>> Error al cargar TPU Coral: {e}. Pasando a CPU...")
    
    interp = tflite.Interpreter(model_path=MODEL_CPU_PATH)
    interp.allocate_tensors()
    print(">>> Modelo cargado en CPU.")
    return interp, False

interpreter, is_tpu = create_interpreter(use_tpu=True)

# ---------------------------------------------------------
# 3. Bucle de la Cámara con Picamera2
# ---------------------------------------------------------
def camera_loop():
    global latest_emotion, current_frame, interpreter, is_tpu
    
    # Inicializar el detector facial
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    # Inicializar la cámara usando la API de Bookworm
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    while True:
        # Capturar el fotograma como un array BGR compatible con OpenCV
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            
            if channels == 1:
                roi_prep = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_prep = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
            roi_prep = cv2.resize(roi_prep, (width, height))
            input_data = np.expand_dims(roi_prep, axis=0)

            if input_details[0]['dtype'] == np.uint8:
                input_data = input_data.astype(np.uint8)
            elif input_details[0]['dtype'] == np.int8:
                input_data = (input_data - 128).astype(np.int8)
            else:
                input_data = (input_data / 255.0).astype(np.float32)

            try:
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                inference_time = (time.time() - start_time) * 1000

                prediction_idx = np.argmax(output_data[0])
                latest_emotion = CLASSES[prediction_idx]

                mode_label = "TPU" if is_tpu else "CPU"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{latest_emotion} [{mode_label}] ({inference_time:.1f}ms)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                if is_tpu:
                    print(f"\n[ALERTA] Fallo TPU ({e}). Conmutando a CPU...")
                    interpreter, is_tpu = create_interpreter(use_tpu=False)

        # Codificar fotograma para la transmisión HTTP
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with lock:
                current_frame = buffer.tobytes()

        time.sleep(0.03)

# Iniciar el hilo de la cámara
threading.Thread(target=camera_loop, daemon=True).start()

# ---------------------------------------------------------
# 4. Servidor Flask
# ---------------------------------------------------------
def generate_frames():
    global current_frame
    while True:
        with lock:
            if current_frame is not None:
                frame_bytes = current_frame
            else:
                time.sleep(0.01)
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('emotionai_web.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def get_emotion():
    return jsonify({'emotion': latest_emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
