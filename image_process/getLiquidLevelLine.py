import cv2 as cv
import numpy as np
from image_process.utils.utils import get_line_kbmodel, is_close_line, is_close_all_lines, capture_roi


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


def adapt_houghlinesP(edg_img):
    """
    对图片进行HoughLinesP可能会得到多条线段，而实际我们只需要两条竖线，所以要对获取到的线段进
    行优化
    :return:
    """
    threshold = 20
    minLineLength = 40
    maxLineGap = 20

    while True:
        lines = cv.HoughLinesP(edg_img, rho=1, theta=np.pi / 180, threshold=threshold, minLineLength=minLineLength,
                               maxLineGap=maxLineGap)
        lines = np.array(lines).reshape(-1, 4)
        # 线的斜率若大于一定值，则舍去
        for line in lines:
            k, _ = get_line_kbmodel(line)
            if k > 0.7:
                lines.remove(lines)

        num_lines = lines.shape[0]
        if num_lines < 2:  # 检测到的线段数量小于2时， 降低阈值
            threshold = threshold - 5
        elif num_lines == 2:  # 判断这2条线段是否接近
            if not is_close_line(lines):
                break
            else:
                threshold = threshold - 5
        elif num_lines > 6: # 检测到的线段太多，增加阈值
            threshold = threshold + 5
        else:   # 当线段数量大于2时, 优化只保留2条直线
            lines = is_close_all_lines(lines)
            break

    return lines


def get_liquid_line(src_img):
    """
    获取液位线：y = line_k * x + line_b
    :param src_img:
    :return: line_k——液位线的斜率, line_b——液位线的偏移量
    """
    # 显示源图像
    if __name__ == '__main__':
        cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
        cv.imshow("00_input", src_img)

    # # 提取图中蓝色背景部分
    # hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    # low_hsv = np.array([44, 0, 38])
    # high_hsv = np.array([143, 255, 255])
    # blue_img = cv.inRange(hsv_img, low_hsv, high_hsv)   # 灰度图像
    # if __name__ == '__main__':
    #     cv.namedWindow("01_blue", cv.WINDOW_KEEPRATIO)
    #     cv.imshow("01_blue", blue_img)

    # # 图像平滑
    # fil_img = cv.GaussianBlur(blue_img, (5, 5), 0)
    # if __name__ == '__main__':
    #     cv.namedWindow("02_filter", cv.WINDOW_KEEPRATIO)
    #     cv.imshow("02_filter", fil_img)
    #
    # # 提取轮廓
    # edg_img = cv.Canny(fil_img, 80, 200, apertureSize=3)
    # if __name__ == '__main__':
    #     cv.namedWindow("03_edges", cv.WINDOW_KEEPRATIO)
    #     cv.imshow("03_edges", edg_img)
    #
    # # hough线段检测
    # lines = adapt_houghlinesP(edg_img)
    # print(lines)
    # # lines = cv.HoughLinesP(edg_img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=40, maxLineGap=20)
    # if __name__ == '__main__':
    #     for i in range(lines.shape[0]):
    #         x1, y1, x2, y2 = lines[i]
    #         cv.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv.namedWindow('06_line', cv.WINDOW_KEEPRATIO)
    #         cv.imshow('06_line', src_img)
    #
    # # 只取出液位柱部分
    # roi_img = get_roi(src_img, lines)
    # if __name__ == '__main__':
    #     cv.imwrite('example/roi.png', roi_img)
    #     cv.namedWindow('07_roi', cv.WINDOW_KEEPRATIO)
    #     cv.imshow('07_roi', roi_img)

    # 获得液体住两侧区域
    roi_img = capture_roi(src_img, True)
    if __name__ == '__main__':
        cv.namedWindow("02_roi", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_roi", roi_img)

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
        img_width = src_img.shape[1]
        pt1 = (0, np.int16(line_k * 0 + line_b))
        pt2 = (img_width, np.int16(line_k * img_width + line_b))
        cv.line(roi_img, pt1, pt2, 255, 1)
        cv.namedWindow("10_line", cv.WINDOW_KEEPRATIO)
        cv.imshow("10_line", roi_img)

    return line_k, line_b


if __name__ == '__main__':
    img = cv.imread('example/gear00027.png')
    get_liquid_line(img)

    cv.waitKey()
    cv.destroyAllWindows()
