from ultralytics import YOLO
from PIL import Image
import requests

model = YOLO('Yolo-Weights\\4-corners-40-best.pt')
image = Image.open('cccd.jpg')
result = model.predict(image, conf=0.25)
