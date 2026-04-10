from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
from PIL import Image
import io, base64

app = Flask(__name__)

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD MODEL 1 (MobileNet)
# ----------------------------
model1 = models.mobilenet_v2(weights=None)
model1.classifier[1] = nn.Linear(model1.last_channel, 7)
model1.load_state_dict(torch.load("mobilenet_bbb.pth", map_location=device))
model1.to(device)
model1.eval()

# ----------------------------
# LOAD MODEL 2 (ViT)
# ----------------------------
model2 = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=7)
model2.load_state_dict(torch.load("vit_bbb.pth", map_location=device))
model2.to(device)
model2.eval()

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# CLASSES
# ----------------------------
classes = [
    "Melanocytic Nevus",
    "Melanoma",
    "Benign Keratosis-like Lesions (BKL)",
    "Basal Cell Carcinoma (BCC)",
    "Actinic Keratoses and Intraepithelial Carcinoma (AKIEC)",
    "Vascular Lesions (VASC)",
    "Dermatofibroma (DF)"
]

# ----------------------------
# PREDICT FUNCTION (UPDATED)
# ----------------------------
def predict(model, img):
    with torch.no_grad():
        output = model(img)

    idx = torch.argmax(output, dim=1).item()
    return classes[idx]

# ----------------------------
# ROUTE
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result1 = result2 = None
    image_data = None

    if request.method == "POST":
        file = request.files["image"]
        model_choice = request.form.get("model_choice")

        if file:
            img_bytes = file.read()
            image_data = base64.b64encode(img_bytes).decode("utf-8")

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # MODEL SELECTION
            if model_choice == "mobilenet":
                p = predict(model1, img_tensor)
                result1 = f"MobileNet → {p}"

            elif model_choice == "vit":
                p = predict(model2, img_tensor)
                result2 = f"ViT → {p}"

            elif model_choice == "both":
                p1 = predict(model1, img_tensor)
                p2 = predict(model2, img_tensor)

                result1 = f"MobileNet → {p1}"
                result2 = f"ViT → {p2}"

    return render_template("index.html",
                           result1=result1,
                           result2=result2,
                           image_data=image_data)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)