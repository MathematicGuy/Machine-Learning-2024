import cv2 as cv

#? Read Image
img = cv.imread('Resources/Photos/cat_large.jpg')
cv.imshow('Cat', img)


# Video Frame have 2 dimension x, y. To re-scale we multiply x, y axis to a float number
# then update frame dimension using cv.resize(frame, (x,y), interpolation=cv.INTER_AREA)
def rescaleFrame(frame, scale=0.75):
    #? Available for Images, Video,
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    # interpolate pixel values when resizing an image.
    # Different interpolation methods can affect the quality and performance of the resizing operation.
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)  # opencv resize command


def changeRes(video, width, height):
    #? Only Available for Live Video
    video.set(3, width)
    video.set(4, height)


#? Resize Image
# resized_image = rescaleFrame(img, 0.5)
# cv.imshow('Image', resized_image)


#? Read Video
video = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = video.read()  # if read or not, read video frame by frame
    #? Resize Video
    frame_resize = rescaleFrame(frame, scale=.4)
    cv.imshow('Video Resized', frame_resize)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if letter d is pressed, break the loop
        break

video.release()
cv.destroyAllWindows()

cv.waitKey(0)  # wait for specific deplay for a key to be pressed
