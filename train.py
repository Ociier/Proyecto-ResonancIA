# -*- coding: utf-8 -*-  

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom  # Utilizaremos pydicom para cargar los archivos DICOM
import numpy as np
from torchvision import transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from PIL import Image

# Definir el dispositivo (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para cargar los archivos DICOM
def dicom_to_pil(file_path: str) -> Image.Image:
    ds = pydicom.dcmread(file_path)
    arr = ds.pixel_array.astype(np.float32)

    # Aplicar rescale si existe en el header
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter

    # Quitar dims de tamaño 1 (tu caso: (1,1,88) -> (88,))
    arr = np.squeeze(arr)

    # Si es 3D, elegir una rebanada 2D razonable
    if arr.ndim == 3:
        slice_axis = int(np.argmin(arr.shape))
        mid_idx = arr.shape[slice_axis] // 2
        arr2d = np.take(arr, indices=mid_idx, axis=slice_axis)
    elif arr.ndim == 2:
        arr2d = arr
    elif arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        side = max(side, 1)
        arr2d = np.resize(arr, (side, side))
    else:
        raise ValueError(f"Forma DICOM no soportada: {arr.shape}")

    # Normalizar a 0-255 y convertir a uint8
    arr2d = arr2d - np.min(arr2d)
    maxv = np.max(arr2d)
    if maxv > 0:
        arr2d = arr2d / maxv
    img_u8 = (arr2d * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_u8)

# Dataset personalizado para manejar las imágenes DICOM
class DicomDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Opcional: mapeo de clases por nombre de carpeta
        self.class_map = class_map or {}

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    fpath = os.path.join(root, f)
                    parent = os.path.basename(root)
                    label = self.class_map.get(parent, 0)  # por defecto 0 si no encuentra
                    self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron .dcm en {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = dicom_to_pil(fpath)
        if self.transform:
            img = self.transform(img)
        return img, label

# Transformaciones para PIL
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Redimensionar a un tamaño estándar
    transforms.Grayscale(num_output_channels=3),  # Convertir a 3 canales (duplicando)
    transforms.ToTensor(),
])

# Mapeo de clases
CLASS_MAP = {"Control": 0, "Leve": 1, "Moderado": 2, "Severo": 3}

# Cargar el dataset
dataset = DicomDataset(
    root_dir=r"C:\Users\Usuario\Downloads\Alzheimer1_dataset\ADNI",
    transform=transform,
    class_map=CLASS_MAP
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Modelo: Usamos ResNet-18 para imágenes 2D
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # Para 4 clases
model = model.to(device)

# Optimizer y Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Pasamos por el modelo
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Retropropagación y optimización
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(dataloader)}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "model_weights.pth")
