<<<<<<< HEAD
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load Model Architecture
# ----------------------------
model = models.mobilenet_v2(pretrained=False)

# 8 output classes
model.classifier[1] = nn.Linear(model.last_channel, 7)

model = model.to(device)

# ----------------------------
# Load Saved Weights
# ----------------------------
model_path = r"C:\Users\srima\OneDrive\Desktop\deep_learning_project\mobilenet_8_final_model.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model Loaded Successfully\n")

# ----------------------------
# Transform (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Class Labels (7 Classes)
# ----------------------------
classes = [
    "Nevus (nv)",
    "Melanoma (mel)",
    "Benign Keratosis (bkl)",
    "Basal Cell Carcinoma (bcc)",
    "Actinic Keratosis (akiec)",
    "Vascular Lesion (vasc)",
    "Dermatofibroma (df)",
    "normal(no lesion)"
]

# ----------------------------
# Image Input
# ----------------------------
image_path = input("Enter image path: ")

if not os.path.exists(image_path):
    print("Image not found!")
    exit()

img = Image.open(image_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

idx = torch.argmax(probs).item()
confidence = probs[0][idx].item()

print("\n===== RESULT =====")
print("Prediction :", classes[idx])
print("Confidence :", round(confidence * 100, 2), "%")

print("\nFinal Diagnosis:")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load Model Architecture
# ----------------------------
model = models.mobilenet_v2(pretrained=False)

# 8 output classes
model.classifier[1] = nn.Linear(model.last_channel, 8)

model = model.to(device)

# ----------------------------
# Load Saved Weights
# ----------------------------
model_path = r"C:\Users\srima\OneDrive\Desktop\deep_learning_project\mobilenet_8_final_model.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model Loaded Successfully\n")

# ----------------------------
# Transform (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Class Labels (8 Classes)
# ----------------------------
classes = [
    "Nevus (nv)",
    "Melanoma (mel)",
    "Benign Keratosis (bkl)",
    "Basal Cell Carcinoma (bcc)",
    "Actinic Keratosis (akiec)",
    "Vascular Lesion (vasc)",
    "Dermatofibroma (df)",
    "normal(no lesion)"
]

# ----------------------------
# Image Input
# ----------------------------
image_path = input("Enter image path: ")

if not os.path.exists(image_path):
    print("Image not found!")
    exit()

img = Image.open(image_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

idx = torch.argmax(probs).item()
confidence = probs[0][idx].item()

print("\n===== RESULT =====")
print("Prediction :", classes[idx])
print("Confidence :", round(confidence * 100, 2), "%")

print("\nFinal Diagnosis:")
>>>>>>> 21d85b86327f2a63b895ddfcb76b10c22a7dfb6f
print("Lesion Type :", classes[idx])