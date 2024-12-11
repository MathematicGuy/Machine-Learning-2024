from concurrent.futures import ThreadPoolExecutor
from skimage import exposure, util
from skimage.filters import rank
from skimage.morphology import disk
import cv2
import os

input_folder = 'unassigned'
output_folder = 'processed_histogram_equalization2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_image(filename):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rescale = exposure.equalize_hist(gray)
    img_rescale_uint8 = util.img_as_ubyte(img_rescale)
    img_eq = rank.equalize(img_rescale_uint8, footprint=disk(25))  # Reduced disk size
    img_resized = cv2.resize(img_eq, (740, int(740 * img_eq.shape[0] / img_eq.shape[1])), interpolation=cv2.INTER_AREA)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_resized)

# Parallelize processing
with ThreadPoolExecutor() as executor:
    executor.map(process_image, [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

print("Processing complete.")
