#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# install.sh — EmotionAI Web
# Instala todas las dependencias en Raspberry Pi OS (Bullseye / Bookworm)
# Uso:
#   chmod +x install.sh
#   ./install.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e   # abortar si cualquier comando falla

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 0. Verificar que no es root puro ─────────────────────────────────────────
if [[ "$EUID" -eq 0 && -z "$SUDO_USER" ]]; then
  warn "Ejecutá como usuario normal con sudo, no directamente como root."
fi

info "=== EmotionAI Web — Instalador ==="
info "Sistema: $(uname -a)"

# ── 1. Actualizar sistema ─────────────────────────────────────────────────────
info "Actualizando lista de paquetes..."
sudo apt-get update -qq

# ── 2. Dependencias del sistema ───────────────────────────────────────────────
info "Instalando dependencias del sistema..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenblas-dev \
    git \
    curl \
    wget \
    usbutils

# ── 3. Entorno virtual Python ─────────────────────────────────────────────────
info "Creando entorno virtual Python..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# ── 4. Pip actualizado ────────────────────────────────────────────────────────
pip install --upgrade pip wheel setuptools -q

# ── 5. TFLite runtime ────────────────────────────────────────────────────────
info "Instalando TFLite runtime..."
pip install tflite-runtime -q || warn "TFLite runtime falló; el servidor usará tensorflow si está instalado."

# ── 6. Flask y dependencias Python ───────────────────────────────────────────
info "Instalando dependencias Python..."
pip install flask flask-cors numpy Pillow opencv-python-headless -q

# ── 7. Google Coral Edge TPU ──────────────────────────────────────────────────
info "Configurando repositorio de Google Coral..."
CORAL_LIST="/etc/apt/sources.list.d/coral-edgetpu.list"
if [[ ! -f "$CORAL_LIST" ]]; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      | sudo tee "$CORAL_LIST"
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | sudo apt-key add -
    sudo apt-get update -qq
fi

info "Instalando libedgetpu1-std (velocidad estándar, menor temperatura)..."
sudo apt-get install -y libedgetpu1-std python3-pycoral || \
  warn "No se pudo instalar pycoral. Podés hacerlo manualmente luego."

# pycoral también disponible vía pip para algunos entornos
pip install pycoral -q || warn "pip install pycoral falló — usando el de apt si está disponible."

# ── 8. Carpeta de modelos ─────────────────────────────────────────────────────
mkdir -p models
if [[ ! -f "models/emociones_edgetpu.tflite" ]]; then
  warn "Modelo Edge TPU no encontrado en models/emociones_edgetpu.tflite"
  info "Podés descargar un modelo preentrenado de ejemplo:"
  info "  https://coral.ai/models/all/"
  info "O colocar tu propio modelo exportado con Edge TPU Compiler."

  info "Descargando modelo FER2013 de ejemplo desde GitHub (si hay conexión)..."
  MODEL_URL="https://raw.githubusercontent.com/niconielsen32/EmotionDetection/main/model/model.tflite"
  wget -q "$MODEL_URL" -O "models/emociones.tflite" && \
    info "Modelo CPU descargado en models/emociones.tflite" || \
    warn "No se pudo descargar el modelo. Colocá uno manualmente."
fi

# ── 9. Habilitar cámara ───────────────────────────────────────────────────────
info "Habilitando interfaz de cámara..."
sudo raspi-config nonint do_camera 0 2>/dev/null || warn "No se pudo habilitar la cámara automáticamente. Hacelo manualmente en raspi-config."

# ── 10. Verificar Coral USB ───────────────────────────────────────────────────
info "Verificando dispositivo Coral USB..."
if lsusb | grep -qi "Google"; then
  info "✅ Coral USB Accelerator detectado"
else
  warn "Coral USB Accelerator no detectado. Asegurate de conectarlo antes de iniciar."
fi

# ── 11. Servicio systemd (opcional) ──────────────────────────────────────────
PROJECT_DIR="$(pwd)"
SERVICE_FILE="/etc/systemd/system/emotionai.service"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

info "Creando servicio systemd para arranque automático..."
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=EmotionAI Web — Detector de Emociones con Coral TPU
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable emotionai
info "Servicio systemd registrado como 'emotionai'"
info "  Iniciar:  sudo systemctl start emotionai"
info "  Ver logs: journalctl -u emotionai -f"

# ── 12. Resumen final ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Instalación completa${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo ""
echo "  Pasos siguientes:"
echo "  1. Colocá tu modelo en:  models/emociones_edgetpu.tflite"
echo "  2. Arrancá el servidor:  source venv/bin/activate && python app.py"
echo "  3. Abrí en el navegador: http://$(hostname -I | awk '{print $1}'):5000"
echo "  4. O usá el HTML standalone apuntando a esa IP."
echo ""
echo "  Si el modelo Edge TPU no está disponible, el servidor"
echo "  arranca automáticamente en modo CPU o modo Demo."
echo ""
