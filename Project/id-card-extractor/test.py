from ultralytics import YOLO
import cv2
from PIL import Image

# Load a model
model = YOLO("weights/best.pt")

# Open a video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale image back to BGR format for YOLO model
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Perform object detection on the grayscale image
    results = model.predict(source=gray_frame_bgr, show=True)

    # Display the resulting frame
    cv2.imshow('frame', gray_frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

