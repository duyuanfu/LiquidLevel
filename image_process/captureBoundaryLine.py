import cv2 as cv
import numpy as np
import copy


# 根据四个点坐标排列出左上右上右下左下位置关系
def sort_four_points(box):
    center_point = np.average(box, axis=0)

    # 分成上下两部分
    top_point = np.zeros((2, 2))
    bottom_point = np.zeros((2, 2))
    top_index = 0
    bottom_index = 0
    for i in range(4):
        if box[i, 1] < center_point[1]:
            top_point[top_index] = box[i]
            top_index = top_index + 1
        else:
            bottom_point[bottom_index] = box[i]
            bottom_index = bottom_index + 1
    # 将上，下两部分点分成左右4部分
    topleft_point = top_point[0] if top_point[0, 0] < center_point[0] else top_point[1]
    topright_point = top_point[1] if top_point[0, 0] < center_point[0] else top_point[0]
    bottomleft_point = bottom_point[0] if bottom_point[0, 0] < center_point[0] else bottom_point[1]
    bottomright_point = bottom_point[1] if bottom_point[0, 0] < center_point[0] else bottom_point[0]

    return np.array([topleft_point, topright_point, bottomleft_point, bottomright_point])


def get_boundary_line(source_img):
    # 读取图像
    src = source_img
    cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
    cv.imshow("00_input", src)

    # 提取图中红色部分
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv.inRange(hsv, low_hsv, high_hsv)   # 灰度图像
    cv.namedWindow("01_mask", cv.WINDOW_KEEPRATIO)
    cv.imshow("01_mask", mask)

    # 图像消噪
    denoise_img = cv.medianBlur(mask, 3)
    cv.namedWindow("02_denoise_img", cv.WINDOW_KEEPRATIO)
    cv.imshow("02_denoise_img", denoise_img)


    # 图像形态转换
    kernel = np.ones((7, 7), np.uint8)
    dilation_img = cv.dilate(denoise_img, kernel, iterations=1)
    cv.namedWindow("03_dilation_img", cv.WINDOW_KEEPRATIO)
    cv.imshow("03_dilation_img", dilation_img)

    # 提取标记红线的区域
    ret, binary_img = cv.threshold(dilation_img, 127, 255, cv.THRESH_BINARY)
    contour_img, contours, hierarchy = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    part_area = [cv.contourArea(contours[i]) for i in range(len(contours))]
    part_area_copy = copy.deepcopy(part_area)
    # 求图像中除了2个轮廓面积最大部分之外的所有部分的轮廓面积和索引
    min_area_value = []
    min_area_index = []
    max_area_value = max(part_area_copy)
    for i in range(len(contours) - 2):
        value = min(part_area_copy)
        index = part_area_copy.index(value)
        # print(value, index)
        part_area_copy[index] = max_area_value
        cv.drawContours(contour_img, [contours[index]], 0, 0, -1)
        # cv.namedWindow("temp_contour_img" + str(i), cv.WINDOW_KEEPRATIO)
        # cv.imshow("temp_contour_img" + str(i), contour_img)

    cv.namedWindow("04_contour_img", cv.WINDOW_KEEPRATIO)
    cv.imshow("04_contour_img", contour_img)

    #     min_area_value.append(value)
    #     min_area_index.append(index)
    # part_area_copy = []
    # print(part_area)
    # print(min_area_value)
    # print(min_area_index)


    # 长方形拟合标记红线
    fit_img, contours, hierarchy = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boxes = np.zeros((2, 4, 2))
    for i in range(2):
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes[i] = box
        cv.drawContours(fit_img, [box], 0, (255, 0, 0), 2)

    cv.namedWindow("05_fit_img", cv.WINDOW_KEEPRATIO)
    cv.imshow("05_fit_img", fit_img)


    # 取长方形上下边的中点
    top_center_point = np.zeros((2, 2))
    bottom_center_point = np.zeros((2, 2))

    for i in range(2):
        boxes[i] = sort_four_points(boxes[i])
        top_center_point[i] = (boxes[i, 0] + boxes[i, 1]) / 2
        bottom_center_point[i] = (boxes[i, 2] + boxes[i, 3]) / 2

    # print("两个正方形的左上，右上，左下，右下点坐标：")
    # print(boxes)
    # print("正方形上边中点的坐标：")
    # print(top_center_point)
    # print("正方形下边中点的坐标：")
    # print(bottom_center_point)

    # 绘制边界线段
    color_img = cv.cvtColor(fit_img, cv.COLOR_GRAY2BGR)
    cv.line(color_img, tuple(np.int0(top_center_point[0]).tolist()), tuple(np.int0(top_center_point[1]).tolist()), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(color_img, tuple(np.int0(bottom_center_point[0]).tolist()), tuple(np.int0(bottom_center_point[1]).tolist()), (0, 255, 0), 2, cv.LINE_AA)
    cv.namedWindow("06_line_img", cv.WINDOW_KEEPRATIO)
    cv.imshow("06_line_img", color_img)


    cv.waitKey(0)
    cv.destroyAllWindows()

    return top_center_point, bottom_center_point


def test(source_img):
    # 读取图像
    src = source_img
    cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
    cv.imshow("00_input", src)

    # 提取图中红色部分
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv.inRange(hsv, low_hsv, high_hsv)   # 灰度图像
    cv.namedWindow("01_mask", cv.WINDOW_KEEPRATIO)
    cv.imshow("01_mask", mask)


if __name__ == '__main__':
    img = cv.imread('example/00.jpg')

    cv.waitKey(0)
    cv.destroyAllWindows()