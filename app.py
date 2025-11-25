import os
import io
import base64

from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

CANDIDATE_MODEL_PATHS = [
    "faces_recog/emotion_model.pth",
    "faces_recog/model/emotion_model.pth",
    "../emotion_model.pth",
    "emotion_model.pth",
    r"C:/Users/jahnavi.ch/Documents/Facial Recognition/emotion_model.pth"
]

def find_model_path():
    for p in CANDIDATE_MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find emotion_model.pth. Looked for:\n" +
        "\n".join(CANDIDATE_MODEL_PATHS)
    )

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*12*12, 256), nn.ReLU(),
            nn.Linear(256, 1)  # placeholder, will be replaced by correct out size below
        )

    def forward(self, x):
        return self.net(x)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

if "model_state" not in checkpoint or "classes" not in checkpoint:
    raise KeyError(f"Checkpoint at {MODEL_PATH} does not contain expected keys "
                   "(need 'model_state' and 'classes').")

classes = checkpoint["classes"]
num_classes = len(classes)

class EmotionCNN_Load(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN_Load, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*12*12, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = EmotionCNN_Load(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"Loaded model from: {MODEL_PATH}")
print("Classes:", classes)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

app = Flask(__name__, template_folder="templates", static_folder="static")

def predict_pil_image(pil_img):
    """
    Input: PIL.Image (RGB or L)
    Output: predicted class (string)
    """
    img = pil_img.convert("L")  # ensure grayscale
    img = transform(img).unsqueeze(0)  # (1,1,48,48)
    with torch.no_grad():
        out = model(img)
        # out shape (1, num_classes)
        pred = torch.argmax(out, dim=1).item()
    return classes[pred]

@app.route("/")
def index():
    # loads templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_upload():
    # expects form upload (file)
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    emotion = predict_pil_image(pil_img)
    return jsonify({"emotion": emotion})

@app.route("/capture", methods=["POST"])
def capture():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,...." }
    """
    data = request.json.get("image") if request.is_json else None
    if not data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        header, encoded = data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Error decoding image: {e}"}), 400

    emotion = predict_pil_image(pil_img)
    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    # use host=0.0.0.0 if you want LAN access, for local testing default is fine
    app.run(debug=True)

