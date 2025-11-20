import streamlit as st
from PIL import Image
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title('ResonancIA')
st.write('Esta aplicación predice Alzheimer a partir de resonancias MRI, comparando tres modelos distintos')

MODEL_DIR = 'Src'

# --- Class Labels ---
CLASS_LABELS = [
    "Mild Alzheimer's Disease",
    "Moderate Alzheimer's Disease",
    "Non-demented",
    "Very Mild Alzheimer's Disease"
]

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
    model1 = models.resnet50(pretrained=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 4) 
    model1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "alzheimer_cnn_model.pth"), map_location=torch.device('cpu')), strict=False)
    model1.eval()
    st.success("Modelo ResNet Cargado")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False


try:
    densenet = models.densenet121(pretrained=False)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, 4) 
    densenet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "alzheimer_DenseNet121_model.pth"), map_location="cpu"))
    densenet.eval()
    st.success("Modelo DenseNet Cargado")
except Exception as e:
    st.warning(f"DenseNet121 no cargado (opcional): {e}")
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

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to("cpu"), y.to("cpu")
            out = model(x)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return precision, recall, cm

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

st.write('Por favor, carga una imagen MRI.')

# Load image
uploaded_file = st.file_uploader("Elegir una imagen MRI", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='MRI Cargado.', use_column_width=True)
    st.write("Clasificando...")
    tensor = preprocess(image)
    label1, conf1, probs1 = predict(model1, tensor)
    label2, conf2, probs2 = predict(model, tensor)
    label3, conf3, probs3 = predict(densenet , tensor)
    labels = ['Mild AD', 'Moderate AD', 'Non-Demented', 'Very Mild AD']
    st.markdown("### Resultados de los modelos")
    st.table({
        "Model": ["ResNet", "EfficientNet", "DenseNet"],
        "Prediction": [labels[label1], labels[label2], labels[label3]],
    })
    st.markdown("## Métricas Detalladas por Modelo")
    VAL_DIR = os.path.join(MODEL_DIR, "Validation")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    models_eval = {
        "ResNet50": model1,
        "EfficientNet-B0": model,
        "DenseNet121": densenet
    }
    for name, model_ref in models_eval.items():
        st.markdown(f"### {name}")

        precision, recall, cm = evaluate_model(model_ref, val_loader, CLASS_LABELS)

        metrics_table = {
            "Clase": CLASS_LABELS,
            "Precision": np.round(precision, 3),
            "Recall": np.round(recall, 3)
        }
        st.table(metrics_table)

        st.markdown("**Matriz de Confusión:**")
        plot_confusion_matrix(cm, CLASS_LABELS)
