from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLOv10
import cv2
import numpy as np
import os
from starlette.responses import RedirectResponse
from starlette.requests import Request
from typing import List

app = FastAPI()
# Load custom weights for YOLOv10
model = YOLOv10('Yolo-Weights/4-corners-best.pt')

UPLOAD_FOLDER = 'images/uploads'
DOWNLOAD_FOLDER = 'images/downloads'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_class=HTMLResponse)
async def detect_files(request: Request, files: List[UploadFile] = File(...)):
    if not files:
        return RedirectResponse(url="/", status_code=303)
    
    result_filenames = []
    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        results = model(filepath)
        result_img = results[0].plot()
        result_filename = 'result_' + file.filename
        result_path = os.path.join(DOWNLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, result_img)
        result_filenames.append(result_filename)
    
    return templates.TemplateResponse("result.html", {"request": request, "filenames": result_filenames})

@app.get("/uploads/{filename}", response_class=HTMLResponse)
async def uploaded_file(request: Request, filename: str):
    return templates.TemplateResponse("result.html", {"request": request, "filename": filename})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")