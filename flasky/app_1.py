from flask import Flask, render_template, Response
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch

app = Flask(__name__)

# Suppress warning messages
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define model identifier
model_identifier = "DrishtiSharma/finetuned-ViT-human-action-recognition-v1"

# Load processor
processor = AutoImageProcessor.from_pretrained(model_identifier)

# Load model
model = AutoModelForImageClassification.from_pretrained(model_identifier)

# Get the class labels
class_labels = model.config.id2label

# Function to perform inference on each frame
def perform_inference(frame):
    # Convert the frame to PIL Image
    image = Image.fromarray(frame)

    # Convert the image to RGB format
    image = image.convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    outputs = model(**inputs)

    # Get predicted class scores
    class_scores = torch.softmax(outputs.logits, dim=-1).squeeze()

    # Sort class scores and get top 5 indices
    top5_indices = class_scores.argsort(descending=True)[:5]

    # Filter out top 5 indices that are valid
    valid_top5_indices = [idx for idx in top5_indices if idx.item() in class_labels]

    # Prepare the predicted classes and their scores
    predicted_classes = []
    for idx in valid_top5_indices:
        class_label = class_labels[idx.item()]
        score = class_scores[idx].item()
        predicted_classes.append((class_label, score))

    return predicted_classes

def video_feed():
    # Open camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference on the frame
        predicted_classes = perform_inference(frame)

        # Display the top predicted classes and their scores on the frame
        y = 20
        for class_label, score in predicted_classes:
            text = f"{class_label}: {score:.4f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 20

        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
