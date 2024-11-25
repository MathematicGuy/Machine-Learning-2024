from ultralytics import YOLO
import cv2 as cv
import cvzone

cap = cv.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n")

