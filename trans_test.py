import torch
import timm
from torchvision import transforms
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
model = timm.create_model(
    "vit_tiny_patch16_224",   # SAME MODEL USED DURING TRAINING
    pretrained=False,
    num_classes=7
)

model = model.to(device)

# ----------------------------
# Load Saved Model
# ----------------------------
model_path = "vit_tiny_final.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Transformer model loaded successfully")

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ----------------------------
# Class Labels
# ----------------------------
classes = [
    "Nevus (nv)",
    "Melanoma (mel)",
    "Benign Keratosis (bkl)",
    "Basal Cell Carcinoma (bcc)",
    "Actinic Keratosis (akiec)",
    "Vascular Lesion (vasc)",
    "Dermatofibroma (df)",
    
]

# ----------------------------
# Input Image
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
print("Confidence :", round(confidence*100,2), "%")