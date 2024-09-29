import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')  # 0: black, 1: white
# cv.imshow('Blank', blank)

# 1. Insert color to blank image 
#?> blank[y1:y2, x1:x2] - height, width
blank[100:150, 100:600] = 0, 0, 255  # draw in a certain range
cv.imshow('Red', blank)


# 2. Draw a rectangle
#?> rectange(image, start_point, end_point, color, thickness)
cv.rectangle(blank, (0, 0), (blank.shape[0]//2, blank.shape[1]//2), (0, 255, 0), thickness=cv.FILLED)
# cv.imshow('Rectangle', blank)


# 3. Draw a circle
#?> circle(image, center, radius, color, thickness)
cv.circle(blank, (blank.shape[0]//2, blank.shape[1]//2), 40, (0, 0, 255), thickness=2)
# cv.imshow('Circle', blank)

# 4. Draw a Line
#?> line(image, start_coordinate, end_coordinate, color, thickness)
cv.line(blank, (100, 100), (200, 200), (255, 0, 0), thickness=3)
# cv.imshow('Line', blank)

# 5. Write Text
#?> putText(image, text, start_coordinate, font, font_scale, color, thickness)
# cv.putText(blank, 'Helo my name is Thanh', (45, 325), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255,0), 1)
# cv.imshow('Text', blank)



cv.waitKey(0)
