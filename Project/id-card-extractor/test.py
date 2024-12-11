import cv2
from PIL import Image
from skimage import exposure, util
from skimage.morphology import disk
from skimage.filters import rank
from ultralytics import YOLO

# Load a model
model = YOLO("weights/best.pt")

# Open a video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        img_rescale = exposure.equalize_hist(gray_frame)
        img_rescale_uint8 = util.img_as_ubyte(img_rescale)
        img_eq = rank.equalize(img_rescale_uint8, footprint=disk(25))  # Reduced disk size

        # Convert equalized grayscale image back to BGR format for YOLO model
        gray_frame_bgr = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)

        # Perform object detection on the grayscale image
        results = model.predict(source=gray_frame_bgr, show=True)

        # Display the resulting frame
        cv2.imshow('frame', gray_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()