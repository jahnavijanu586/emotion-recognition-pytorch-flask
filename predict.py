import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

MODEL_PATH = r"C:/Users/jahnavi.ch/Documents/Facial Recognition/emotion_model.pth"

CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print("Loaded Model Classes:", CLASSES)

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionCNN, self).__init__()
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

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model = EmotionCNN(num_classes=6)
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def predict(image_path):
    if not os.path.exists(image_path):
        print(" ERROR: File not found →", image_path)
        return

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    print("\nPredicted Emotion:", CLASSES[predicted.item()])

if __name__ == "__main__":
    image_path = input("Enter image path → ")
    predict(image_path)
