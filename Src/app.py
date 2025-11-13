import streamlit as st
from PIL import Image
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from efficientnet_pytorch import EfficientNet

st.title('ResonancIA')
st.write('Esta aplicación predice Alzheimer a partir de resonancias MRI, comparando tres modelos distintos')

MODEL_DIR = 'Src'
MODELS = {
    "EfficientNet-B0": os.path.join(MODEL_DIR, "alzheimer_efficientnet_model.pth"),
    "Custom CNN": os.path.join(MODEL_DIR, "alzheimer_cnn_model.pth")
}

# --- Class Labels ---
CLASS_LABELS = [
    "Mild Alzheimer's Disease",
    "Moderate Alzheimer's Disease",
    "Non-demented",
    "Very Mild Alzheimer's Disease"
]

class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

try:
    # Load model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "alzheimer_efficientnet_model.pth"), map_location=torch.device('cpu')))
    model.eval()
    model_loaded = True
    st.success("Modelo EfficientNet Cargado")
except FileNotFoundError:
    st.error(f"Model file not found. Please check the path.")
    model_loaded = False

try:
    model1 = AlzheimerCNN()
    model1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "alzheimer_cnn_model.pth"), map_location=torch.device('cpu')))
    model1.eval()
    model_loaded = True
    st.success("Modelo CNN Cargado")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Preprocess image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Predict
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs.squeeze().tolist()


st.write('Por favor, carga una imagen MRI.')

# Load image
uploaded_file = st.file_uploader("Elegir una imagen MRI", type=["jpg", "jpeg"])
if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='MRI Cargado.', use_column_width=True)
    st.write("Clasificando...")
    tensor = preprocess(image)
    label1, conf1, probs1 = predict(model1, tensor)
    label2, conf2, probs2 = predict(model, tensor)
    labels = ['Mild AD', 'Moderate AD', 'Non-Demented', 'Very Mild AD']
    st.markdown("### Resultados de los modelos")
    st.table({
        "Model": ["Modelo CNN", "EfficientNet"],
        "Prediction": [labels[label1], labels[label2]],
    })