import cv2 as cv
import os

print('current dir:', os.getcwd())

#? Read Image
#! Limitation: can't read large image effectively
# img = cv.imread('Resources/Photos/cat_large.jpg')
# cv.imshow('Cat', img)

#? Read Video
video = cv.VideoCapture('Resources/Videos/dog.mp4')
while True:
    isTrue, frame = video.read()  # if read or not, read video frame by frame
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if letter d is pressed, break the loop
        break

video.release()
cv.destroyAllWindows()

cv.waitKey(0)  # wait for specific deplay for a key to be pressed
