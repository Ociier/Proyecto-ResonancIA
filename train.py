# -*- coding: utf-8 -*-
"""
Entrenamiento del modelo ResNet-18 con DICOMs de Alzheimer
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
from PIL import Image

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def dicom_to_pil(file_path: str) -> Image.Image:
    """Convierte un archivo DICOM a imagen PIL RGB."""
    ds = pydicom.dcmread(file_path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter
    arr = np.squeeze(arr)

    # Elegir una rebanada central si es 3D
    if arr.ndim == 3:
        slice_axis = int(np.argmin(arr.shape))
        mid_idx = arr.shape[slice_axis] // 2
        arr2d = np.take(arr, indices=mid_idx, axis=slice_axis)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        side = int(np.sqrt(arr.size))
        arr2d = np.resize(arr, (side, side))

    # Normalizar a 0–255 y convertir a uint8
    arr2d -= np.min(arr2d)
    maxv = np.max(arr2d)
    if maxv > 0:
        arr2d /= maxv
    img_u8 = (arr2d * 255.0).astype(np.uint8)

    return Image.fromarray(img_u8).convert("RGB")

# -----------------------------
# DATASET PERSONALIZADO
# -----------------------------
class DicomDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = class_map or {}

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(".dcm"):
                    fpath = os.path.join(root, f)
                    parent = os.path.basename(root)
                    # Buscar clase según nombre del folder
                    label = None
                    for k, v in self.class_map.items():
                        if k.lower() in parent.lower():
                            label = v
                            break
                    if label is not None:
                        self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron archivos DICOM en {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = dicom_to_pil(fpath)
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# TRANSFORMACIONES
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# MAPEOS DE CLASES (4 reales)
# -----------------------------
CLASS_MAP = {
    "Control": 0,
    "Leve": 1,
    "Moderado": 2,
    "Severo": 3
}
CLASS_NAMES = ["Control", "Leve", "Moderado", "Severo"]

# -----------------------------
# CARGAR DATOS
# -----------------------------
dataset = DicomDataset(
    root_dir=r"C:\Users\Usuario\Downloads\Alzheimer1_dataset\ADNI",
    transform=transform,
    class_map=CLASS_MAP
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
print(f"📂 Total de muestras: {len(dataset)}")

# -----------------------------
# MODELO
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # 🔹 4 clases
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# ENTRENAMIENTO
# -----------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(dataloader):.6f}")

# -----------------------------
# GUARDAR PESOS
# -----------------------------
torch.save(model.state_dict(), "model_weights.pth")
print("✅ Modelo entrenado y guardado en 'model_weights.pth'")
