import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

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

# Load YOLOv3 object detection model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to perform person detection
def detect_person(frame):
    # Get frame dimensions
    (H, W) = frame.shape[:2]

    # Determine the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Create a blob from the frame and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = torch.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to 'person'
                # Scale the bounding box coordinates to the original image size
                box = detection[0:4] * torch.tensor([W, H, W, H])
                (centerX, centerY, width, height) = box.int().tolist()

                # Calculate the top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw bounding boxes and labels on the frame
    for i in indices:
        i = i[0]
        (x, y, w, h) = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

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

    # Filter out top 5 indices that are valid and have scores greater than 40
    valid_top5_indices = [idx for idx in top5_indices if idx.item() in class_labels and class_scores[idx].item() > 40]

    # Prepare the predicted classes and their scores
    predicted_classes = []
    for idx in valid_top5_indices:
        class_label = class_labels[idx.item()]
        score = class_scores[idx].item()
        predicted_classes.append((class_label, score))

    return predicted_classes

# Open camera
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to capture frames and perform inference
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect persons in the frame
    frame = detect_person(frame)

    # Perform action recognition on the detected person
    predicted_classes = perform_inference(frame)

    # Display the top predicted classes and their scores on the frame
    y = 20
    for class_label, score in predicted_classes:
        text = f"{class_label}: {score:.4f}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 20

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
