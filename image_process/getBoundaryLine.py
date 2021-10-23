import cv2 as cv
import numpy as np


def conv2d(img, kernel):
    """
    二维卷积
    :param img:
    :param kernel:
    :return:
    """
    (height, width) = img.shape
    ksize = kernel.shape[0]

    for i in range(height - ksize):
        for j in range(width - ksize):
            img[i, j] = np.sum(img[i: i + ksize, j: j + ksize] * kernel)

    return img


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


def cal_four_point_xy(source_img):
    """
    获取四个边界点
    :param source_img: 输入RGB图像
    :return: center_xy = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    # 读取图像
    src = source_img
    if __name__ == '__main__':
        cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
        cv.imshow("00_input", src)

    # 提取图中蓝色背景部分
    hsv_img = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    low_hsv = np.array([70, 6, 38])
    high_hsv = np.array([143, 255, 255])
    mask_img = cv.cv2.inRange(hsv_img, low_hsv, high_hsv)   # 灰度图像
    if __name__ == '__main__':
        cv.namedWindow("01_mask", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_mask", mask_img)

    # 图像平滑
    fil_img = cv.GaussianBlur(mask_img, (5, 5), 0)
    if __name__ == '__main__':
        cv.namedWindow("02_filter", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_filter", fil_img)

    # 提取轮廓
    edg_img = cv.Canny(fil_img, 80, 200, apertureSize=3)
    if __name__ == '__main__':
        cv.namedWindow("03_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_edges", edg_img)

    # 闭运算
    kernel = np.ones((9, 9), np.uint8)
    clo_img = cv.morphologyEx(edg_img, cv.MORPH_CLOSE, kernel)
    if __name__ == '__main__':
        cv.namedWindow("04_close", cv.WINDOW_KEEPRATIO)
        cv.imshow("04_close", clo_img)

    # 中值滤波
    blr_img = cv.medianBlur(clo_img, ksize=7)
    if __name__ == '__main__':
        cv.namedWindow("05_blur", cv.WINDOW_KEEPRATIO)
        cv.imshow("05_blur", blr_img)

    # 选定4个最大轮廓，排除其他噪点
    ret, thresh = cv.threshold(blr_img, 127, 255, 0)
    cnt_img, contours, hierarchy = cv.findContours(thresh, 1, 2)

    part_area = [cv.contourArea(contours[i]) for i in range(len(contours))]
    max_area_index = cal_largest_value(part_area, 4)    # 获取面积最大四部分的轮廓
    for i in range(len(part_area)):
        if i not in max_area_index:     # 将其余噪点的灰度值置0
            cv.drawContours(cnt_img, [contours[i]], 0, 0, -1)
    if __name__ == '__main__':
        cv.namedWindow("06_contour", cv.WINDOW_KEEPRATIO)
        cv.imshow("06_contour", cnt_img)

    # 计算4个轮廓的质心
    center_xy = []
    for i in range(len(part_area)):
        if i in max_area_index:
            cnt = contours[i]
            M = cv.moments(cnt)     # 获取轮廓的特征矩
            c_x = int(M['m10'] / M['m00'])
            c_y = int(M['m01'] / M['m00'])
            center_xy.append((c_x, c_y))

    if __name__ == '__main__':
        temp_img = cv.cvtColor(cnt_img, cv.COLOR_GRAY2BGR)
        for i in range(4):
            # temp_img[center_xy[i][1], center_xy[i][0]] = (0, 0, 255)
            cv.circle(temp_img, center_xy[i], radius=1, color=(0, 0, 255), thickness=2)

        # cv.line(temp_img, center_xy[0], center_xy[1], (0, 0, 255), 1, cv.LINE_AA)
        # cv.line(temp_img, center_xy[2], center_xy[3], (0, 0, 255), 1, cv.LINE_AA)
        cv.namedWindow("07_center", cv.WINDOW_KEEPRATIO)
        cv.imshow("07_center", temp_img)

    return center_xy


if __name__ == '__main__':
    img = cv.imread('./example/test00.jpg')
    center_xy = cal_four_point_xy(img)

    cv.waitKey()
    cv.destroyAllWindows()