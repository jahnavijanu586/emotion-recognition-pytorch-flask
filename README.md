# emotion-recognition-pytorch-flask
A deep learning–based Facial Emotion Recognition System built using PyTorch and deployed with a Flask web application. Detects emotions like Happy, Sad, Angry, Neutral, Fear, and Surprise from images or webcam captures.

# Features :

1. Detects 6 human emotions

2. Custom CNN built with PyTorch

3. Supports image upload + webcam capture

4. Real-time web application using Flask

5. Preprocessing pipeline using PIL + Torchvision

6. CLI script for direct predictions

7. Modular training, inference, and deployment workflow:

# Tech Stack
Languages:
Python

Libraries:

1.PyTorch

2.Torchvision

3. Flask

4.Pillow

5. Base64 + JSON (for webcam image input)

# Model Overview:
The project uses a Convolutional Neural Network trained on 48×48 grayscale images.

# Model Layers

> Conv2D → ReLU

> Conv2D → ReLU

> MaxPool

> Conv2D → ReLU

> MaxPool

> Flatten

> Fully Connected Layer (256 units)

> Output Layer (number of emotion classes)

# Training the Model
Run:
python train_model.py

What this script does:

>Loads dataset using ImageFolder

> Applies preprocessing

> Trains CNN for 10 epochs

> Saves model to emotion_model.pth

# Predict Emotion (CLI)

Run:
python predict.py

Enter the image path, and output will be:
Predicted Emotion: Happy

# Running the Flask Web App

Start the server:
python app.py

Open in browser:
ttp://127.0.0.1:5000/

> Upload an image

> Capture a photo using webcam

> Get real-time emotion prediction

Example JSON output:

{
  "emotion": "Neutral"
}

# How Image Prediction Works

1.Convert image to grayscale

2. Resize to 48×48

3. Apply transforms

4. Pass image through trained CNN

5. Return highest-probability emotion

# Requirements

Create requirements.txt and include:

flask
torch
torchvision
pillow
numpy

# Future Enhancements

> Add face detection using OpenCV

> Deploy online using Render / AWS / Azure

> Improve accuracy using transfer learning

> Add voice emotion detection

