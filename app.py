from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import pydicom  # Usamos pydicom para cargar los archivos DICOM

app = Flask(__name__)
CORS(app)

# Definir el dispositivo (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Usamos ResNet-18 para 2D
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # Para 4 clases
model = model.to(device)

# Lista de clases
CLASS_NAMES = ['Control', 'Leve', 'Moderado', 'Severo']

# ---------- Funciones para procesamiento de imágenes ----------
def dicom_to_pil(file_path: str) -> Image.Image:
    ds = pydicom.dcmread(file_path)
    arr = ds.pixel_array.astype(np.float32)

    # Aplicar rescale si existe en el header
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter

    # Si es 3D, elegir una rebanada 2D razonable
    if arr.ndim == 3:
        slice_axis = int(np.argmin(arr.shape))
        mid_idx = arr.shape[slice_axis] // 2
        arr2d = np.take(arr, indices=mid_idx, axis=slice_axis)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError(f"Forma DICOM no soportada: {arr.shape}")

    # Normalizar a 0-255 y convertir a uint8
    arr2d = arr2d - np.min(arr2d)
    maxv = np.max(arr2d)
    if maxv > 0:
        arr2d = arr2d / maxv
    img_u8 = (arr2d * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_u8)

# ---------- Helper ----------
def allowed_ext(filename):
    fn = filename.lower()
    return fn.endswith((".dcm"))

def predict_image(file_path):
    fp = file_path.lower()
    if fp.endswith(".dcm"):
        img = dicom_to_pil(file_path)
    else:
        raise ValueError("Formato no soportado. Usa .dcm")

    # Aplicar transformaciones
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Redimensionar a un tamaño estándar
        transforms.Grayscale(num_output_channels=3),  # Convertir a 3 canales (duplicando)
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).to(device)  # Añadir batch dimension y mover a dispositivo

    with torch.no_grad():
        logits = model(img)                   # [1,4]
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES[pred_idx]

# ---------- Rutas ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"device": "cuda" if torch.cuda.is_available() else "cpu", "status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Falta el campo 'file' en multipart/form-data"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400
    if not allowed_ext(f.filename):
        return jsonify({"error": "Formato no soportado. Usa .dcm"}), 415

    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, f.filename)
    f.save(save_path)

    try:
        pred = predict_image(save_path)
        return jsonify({"prediction": pred}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
