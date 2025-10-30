from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image, UnidentifiedImageError
import pydicom
import sys


# -------------------------------
# Configuración base
# -------------------------------
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar arquitectura base
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

# Cargar pesos entrenados (ajusta la ruta si es necesario)
weights_path = "model_weights.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    print(f"✅ Pesos cargados desde {weights_path}")
else:
    print("⚠️ No se encontraron pesos entrenados. Usando modelo no entrenado.")

model.eval()

# Clases
CLASS_NAMES = ["Control", "Leve", "Moderado", "Severo", "No es una resonancia"]

# -------------------------------
# Funciones de lectura de imágenes
# -------------------------------
def dicom_to_pil(file_path: str):
    """Convierte un DICOM a una imagen PIL."""
    try:
        ds = pydicom.dcmread(file_path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + inter

        # Elige una rebanada central si es 3D
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            arr2d = arr[mid]
        elif arr.ndim == 2:
            arr2d = arr
        else:
            raise ValueError(f"Forma DICOM no soportada: {arr.shape}")

        # Normalizar y convertir a imagen
        arr2d -= np.min(arr2d)
        if np.max(arr2d) > 0:
            arr2d /= np.max(arr2d)
        img_u8 = (arr2d * 255).astype(np.uint8)

        return Image.fromarray(img_u8).convert("RGB")
    except Exception as e:
        print(f"Error al leer DICOM: {e}")
        raise ValueError("Archivo DICOM inválido o corrupto")

def standard_image(file_path: str):
    """Carga una imagen JPG/PNG común."""
    try:
        return Image.open(file_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Archivo de imagen no válido")

# -------------------------------
# Helper
# -------------------------------
def allowed_ext(filename):
    fn = filename.lower()
    return fn.endswith((".dcm", ".jpg", ".jpeg", ".png"))

def predict_image(file_path):
    """Predice la clase del archivo cargado."""
    fp = file_path.lower()

    # 1️⃣ Cargar la imagen
    try:
        if fp.endswith(".dcm"):
            img = dicom_to_pil(file_path)
        elif fp.endswith((".jpg", ".jpeg", ".png")):
            img = standard_image(file_path)
        else:
            return "No es una resonancia"
    except Exception:
        return "No es una resonancia"

    # 2️⃣ Transformar para el modelo
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)

    # 🔍 DEPURACIÓN: imprime tamaño y tipo
    print(f"🧠 Ejecutando predicción para: {os.path.basename(file_path)}")
    print(f"   → Tensor shape: {img.shape}")

    # 3️⃣ Inferencia
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs))
        print(f"   → Vector de probabilidades: {probs.cpu().numpy().round(3)}")
        print(f"   → Índice predicho: {pred_idx} ({CLASS_NAMES[pred_idx]}), confianza={conf:.3f}")

    # 4️⃣ Si la confianza es muy baja, marcar como “no resonancia”
    if conf < 0.5:
        print(f"⚠️ Confianza baja ({conf:.2f}) → No es resonancia")
        return "No es una resonancia"

    print(f"✅ Predicción final: {CLASS_NAMES[pred_idx]} (confianza={conf:.2f})")
    import sys
    sys.stdout.flush()

    return CLASS_NAMES[pred_idx]

# -------------------------------
# Rutas Flask
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "status": "ok"
    })

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Falta el campo 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400
    if not allowed_ext(f.filename):
        return jsonify({"prediction": "No es una resonancia"}), 200

    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, f.filename)
    f.save(save_path)

    try:
        pred = predict_image(save_path)
        return jsonify({"prediction": pred}), 200
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Forzar salida inmediata de los prints
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    os.environ["FLASK_ENV"] = "development"

    # Ejecutar sin el auto-reloader (evita doble proceso)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
