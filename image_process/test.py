import cv2 as cv
import numpy as np


def cal_largest_value(list, num_value):
    """
    :param list:
    :param num_value: 将列表元素从大到小排列，取前面num_value个值
    :return:
    """
    max_area_index = []
    for i in range(num_value):
        max_area_value = max(list)
        index = list.index(max_area_value)
        max_area_index.append(index)
        list[index] = 0

    return max_area_index


img = cv.imread('example/edge00.png', 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
cnt_img, contours, hierarchy = cv.findContours(thresh, 1, 2)

print(len(contours))

for i in range(len(contours)):
    cnt = contours[i]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0, 0, 255), 2)

cv.namedWindow('cnt', cv.WINDOW_KEEPRATIO)
cv.imshow('cnt', img)
cv.waitKey()
cv.destroyAllWindows()
