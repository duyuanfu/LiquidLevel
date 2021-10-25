import numpy as np
import cv2 as cv

def remove_small_contours(input_img):
    cnt_img, contours, hierarchy = cv.findContours(input_img, 1, 2)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)



img = cv.imread('image_process/example/edge00.png', 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
cnt_img, contours, hierarchy = cv.findContours(thresh, 1, 2)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for i in range(len(contours)):
    cnt = contours[i]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0, 0, 255), 1)

cv.namedWindow("img", cv.WINDOW_KEEPRATIO)
cv.imshow("img", img)
cv.waitKey()
cv.destroyAllWindows()