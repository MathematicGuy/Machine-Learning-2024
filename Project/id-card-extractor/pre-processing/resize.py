import cv2
import os

# Path to the folder containing images
input_folder = 'processed_histogram_equalization'
output_folder = 'processed_histogram_equalization'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Resize the image to width 640 while maintaining aspect ratio
        height, width = img.shape[:2]
        new_width = 740
        new_height = int((new_width / width) * height)
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_resized)

print("Processing complete.")
