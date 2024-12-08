from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO("../Yolo-Weights/yolov8n")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        results = model(filepath)
        result_img = results[0].plot()
        result_path = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
        cv2.imwrite(result_path, result_img)
        return redirect(url_for('uploaded_file', filename='result_' + file.filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)