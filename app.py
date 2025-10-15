from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18
import nibabel as nib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar el modelo ResNet-18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = r3d_18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases: Control, Leve, Moderado, Severo
model = model.to(device)

# Función para cargar imágenes NIfTI
def load_nifti_image(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    data = np.expand_dims(data, axis=0)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    transform = transforms.Compose([transforms.Resize((64, 64, 64))])
    data = transform(data).to(device)
    return data

# Función para cargar imágenes TGA
def load_tga_image(file_path):
    image = Image.open(file_path)
    data = np.array(image)
    data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=0)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(device)
    return data

# Función de inferencia
def predict_image(file_path):
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        image = load_nifti_image(file_path)
    elif file_path.endswith('.tga'):
        image = load_tga_image(file_path)
    else:
        raise ValueError("Unsupported file format")

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    class_names = ['Control', 'Leve', 'Moderado', 'Severo']
    return class_names[predicted_class.item()]

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)
    predicted_class = predict_image(file_path)
    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
