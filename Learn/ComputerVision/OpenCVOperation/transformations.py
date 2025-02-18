import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/group 1.jpg')

cv.imshow('Boston', img)


#! Translation: translate image relative to the image frame
# x: Right
# y: Down
# -x: Left
# -y: Up
def translate(img, x, y):
    transMatrix = np.float32(([1, 0, x], [0, 1, y]))
    dimensions = (img.shape[1], img.shape[0])  # width, height

    return cv.warpAffine(img, transMatrix, dimensions)


translated = translate(img, 100, -100)
cv.imshow('Translated', translated)

#! Rotation
def rotate(img, angle, rotationPoint=None):
    (height, width) = img.shape[:2]

    if rotationPoint is None:  # assume we're rotating around the center
        rotationPoint = (width // 2, height // 2)

    rotMatrix = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMatrix, dimensions)


# rotated = rotate(img, 45)
# cv.imshow('Rotated', rotated)

# # Flipping: 0: vertical, 1: horizontal, -1: both
# flip = cv.flip(img, -1)
# cv.imshow('Flip', flip)

cv.waitKey(0)