import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat', img)

#! Converting img to greyscale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', grey)

#! Blur
#?> GaussianBlur(image, kernel_size, border_type) larger the kernel_size, the more blur
#?> Can use to reduces edges detection
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

#! Edge Cascade
#?> Canny(image, threshold1, threshold2)
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

#! Dialated Image
#?> dialate(image, kernel, iteration): morphological operation that grows and thickens objects in a binary image.
#?> Can be use to fill up missing holes, minor missing part in a object
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroded
eroded = cv.erode(dilated, (3, 3), iterations=1)  # reverse dialated image
cv.imshow('Eroded', eroded)

# Resize & Crop Image
resized = cv.resize(img, (550, 250), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)