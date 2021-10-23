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


def get_roi(input_img, pt_list):
    """
    提取两条直线之间的区域
    :param img: rgb图像
    :param list_pt: 两条直线4个点的坐标
    :return:
    """
    pt1 = pt_list[0][0]
    pt2 = pt_list[1][0]
    # print(pt1, pt2)

    if pt1[0] == pt1[2]:
        pt1[0] += 1
    k1 = (pt1[3] - pt1[1]) / (pt1[2] - pt1[0])
    b1 = - k1 * pt1[0] + pt1[1]

    if pt2[0] == pt2[2]:
        pt2[0] += 1
    k2 = (pt2[3] - pt2[1]) / (pt2[2] - pt2[0])
    b2 = - k2 * pt2[0] + pt2[1]
    # print(k1, b1, k2, b2)

    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            y1 = k1 * j + b1
            y2 = k2 * j + b2
            if not(y1 >= i >= y2 or y1 <= i <= y2):
                input_img[i, j] = [0, 0, 0]

    return input_img


def get_top_edge(input_img):
    """

    :param input_img: 二值化图像
    :return: top_edge = [(x1, y1), (x2, y2), ...(xn, yn)]上边沿所在点组成的列表
    """
    height, width= input_img.shape
    top_edge = []
    rgb_img = cv.cvtColor(input_img, cv.COLOR_GRAY2BGR)

    for j in range(width):  # 从左往右扫描
        for i in range(height):     # 从上到下扫描
            if input_img[i, j] != 0:
                top_edge.append((j, i))
                rgb_img[i, j] = [0, 0, 255]
                break

    if __name__ == '__main__':
        cv.namedWindow("top_edge", cv.WINDOW_KEEPRATIO)
        cv.imshow("top_edge", rgb_img)

    return top_edge


def get_liquid_line(src_img):
    """
    获取液位线：y = line_k * x + line_b
    :param source_img:
    :return: line_k——液位线的斜率, line_b——液位线的偏移量
    """
    # 显示源图像
    if __name__ == '__main__':
        cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
        cv.imshow("00_input", src_img)

    # 提取图中蓝色背景部分
    hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    low_hsv = np.array([44, 0, 38])
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

    # 提取竖直线
    ver_img = cv.Sobel(edg_img, cv.CV_8U, dx=1, dy=0, ksize=3)
    if __name__ == '__main__':
        cv.namedWindow('04_vertical', cv.WINDOW_KEEPRATIO)
        cv.imshow('04_vertical', ver_img)

    # 闭运算
    kernel = np.ones((3, 3), np.uint8)
    clo_img = cv.morphologyEx(ver_img, cv.MORPH_CLOSE, kernel)
    if __name__ == '__main__':
        cv.namedWindow("05_close", cv.WINDOW_KEEPRATIO)
        cv.imshow("05_close", clo_img)

    # hough线段检测
    lines = cv.HoughLinesP(edg_img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=40, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(lines)
    if __name__ == '__main__':
        cv.namedWindow('06_line', cv.WINDOW_KEEPRATIO)
        cv.imshow('06_line', src_img)

    # 只取出液位柱部分
    roi_img = get_roi(src_img, lines)
    if __name__ == '__main__':
        cv.namedWindow('07_roi', cv.WINDOW_KEEPRATIO)
        cv.imshow('07_roi', roi_img)

    # 提取黄色区域
    hsv = cv.cvtColor(roi_img, cv.COLOR_BGR2HSV)
    low_hsv = np.array([0, 120, 46])
    high_hsv = np.array([40, 255, 255])
    yel_img = cv.inRange(hsv, low_hsv, high_hsv)   # 灰度图像
    if __name__ == '__main__':
        cv.namedWindow("08_mask", cv.WINDOW_KEEPRATIO)
        cv.imshow("08_mask", yel_img)

    # 提取上边沿
    top_edge = get_top_edge(yel_img)
    top_edge = np.array(top_edge).reshape(-1, 2)

    # 将构成上边沿的离散点拟合成一条直线
    line_info = cv.fitLine(top_edge, cv.DIST_FAIR, 0, 0.01, 0.01)
    # y = k * x + b
    line_k = line_info[1] / line_info[0]
    line_b = line_info[3] - line_k * line_info[2]

    if __name__ == '__main__':
        pt1 = (20, np.int16(line_k * 20 + line_b))
        pt2 = (100, np.int16(line_k * 100 + line_b))
        cv.line(roi_img, pt1, pt2, 255, 1)
        cv.namedWindow("10_line", cv.WINDOW_KEEPRATIO)
        cv.imshow("10_line", roi_img)

    return line_k, line_b


if __name__ == '__main__':
    img = cv.imread('example/test01.jpg')
    get_liquid_line(img)

    cv.waitKey()
    cv.destroyAllWindows()
