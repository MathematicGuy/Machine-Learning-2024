from typing import List
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import torch 

app = FastAPI()

# Load the YOLO model with custom weights
try:
    model = YOLO('yolo11n.pt')  # Load the base model first
    model.load_state_dict(torch.load('best.pt')['model'].state_dict()) # Load your custom weights
    print("Loaded custom weights successfully")
except Exception as e:
    print(f"Error loading weights: {e}")
    model = None  # Set model to None if loading fails

# (Optional) Override classes if needed during training.
# model.names = {0: 'person', 1: 'car', ...}

@app.get("/")
async def root():
    return {"message": "YOLOv11n Object Detection API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded properly"}
    # --- Handle the image ---
    # Method 1: Using BytesIO and PIL (Recommended for most cases)
    image = Image.open(BytesIO(await file.read()))
    
    # Method 2: Using OpenCV (For more advanced image processing)
    # contents = await file.read()
    # nparr = np.frombuffer(contents, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- Perform Inference ---
    try:
        results = model(image)[0]
    except Exception as e:
        return {"error": f"Error during inference: {e}"}

    # --- Process Results ---
    # Example: Get bounding boxes, confidence scores, and class IDs
    boxes = results.boxes.xyxy.tolist()
    confidences = results.boxes.conf.tolist()
    class_ids = results.boxes.cls.tolist()

    # --- Convert results to a dictionary ---
    detections = []
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        detections.append({
            "box": box,
            "confidence": confidence,
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)]  # Get class name from model.names
        })

    return {"detections": detections}

@app.post("/predict_batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    if model is None:
        return {"error": "Model not loaded properly"}

    images = []
    for file in files:
        image = Image.open(BytesIO(await file.read()))
        images.append(image)

    try:
        results_list = model(images)
    except Exception as e:
        return {"error": f"Error during inference: {e}"}

    batch_detections = []
    for results in results_list:
        boxes = results.boxes.xyxy.tolist()
        confidences = results.boxes.conf.tolist()
        class_ids = results.boxes.cls.tolist()

        detections = []
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            detections.append({
                "box": box,
                "confidence": confidence,
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)]
            })
        batch_detections.append({"detections": detections})

    return {"batch_detections": batch_detections}