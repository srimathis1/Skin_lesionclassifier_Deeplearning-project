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
model1.load_state_dict(torch.load("mobilenet_final_model.pth", map_location=device))
model1.to(device)
model1.eval()

# ----------------------------
# LOAD MODEL 2 (ViT - timm)
# ----------------------------
model2 = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=7)
model2.load_state_dict(torch.load("vit_tiny_final.pth", map_location=device))
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
    "Nevus", "Melanoma", "BKL",
    "BCC", "AKIEC", "VASC", "DF"
]

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict(model, img):
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    idx = torch.argmax(probs).item()
    confidence = round(probs[0][idx].item()*100, 2)

    return classes[idx], confidence

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

            # ----------------------------
            # MODEL SELECTION LOGIC
            # ----------------------------
            if model_choice == "mobilenet":
                p, c = predict(model1, img_tensor)
                result1 = f"MobileNet → {p} ({c}%)"

            elif model_choice == "vit":
                p, c = predict(model2, img_tensor)
                result2 = f"ViT → {p} ({c}%)"

            elif model_choice == "both":
                p1, c1 = predict(model1, img_tensor)
                p2, c2 = predict(model2, img_tensor)

                result1 = f"MobileNet → {p1} ({c1}%)"
                result2 = f"ViT → {p2} ({c2}%)"

    return render_template("index.html",
                           result1=result1,
                           result2=result2,
                           image_data=image_data)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)