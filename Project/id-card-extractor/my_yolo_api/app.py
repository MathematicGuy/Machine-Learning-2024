from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8n model (modify the path if using a custom model)
model = YOLO("..\weights\\best.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Perform object detection on an uploaded image.
    """
    try:
        # Load the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Run YOLO detection
        results = model(image)

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy.tolist()[0]  # Bounding box coordinates [x1, y1, x2, y2]
                confidence = box.conf.tolist()[0]  # Confidence score
                class_id = int(box.cls.tolist()[0])  # Class ID
                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": bbox
                })

        return JSONResponse(content={"detections": detections})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app using uvicorn (uncomment below to test locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
