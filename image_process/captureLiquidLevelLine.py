import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def find_circles(src):
    """
    找出图像中的圆形
    :param src: RGB图像
    :return: 找到的圆的参数列表
    """
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    circles = cv.HoughCircles(image=blur, method=cv.HOUGH_GRADIENT,
                              dp=1,
                              minDist=20,  # 10 两个圆之间圆心的最小距离.如果太小的，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这参数设置太大，某些圆就不能被检测出来。
                              param1=60,
                              param2=80,
                              minRadius=10,  # 圆半径的最小值
                              maxRadius=250)  # 圆半径的最大值
    circles = np.uint16(np.around(circles))
    return circles


def get_circular_mask(img_size, circle):
    """
    获取指定区域的圆形掩膜,掩膜图像尺寸与输入图像相同
    :param src:
    :param circle:
    :return: mask -获取指定区域的圆形掩膜
    """
    [center_x, center_y, ridius] = circle
    mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    mask = cv.circle(mask, (center_x, center_y), ridius, 255, -1)
    return mask


def extract_circular_regoin(src, mask):
    """
    获取指定的圆形区域
    :param src:
    :param circle:
    :return:
    """
    dst = cv.bitwise_and(src, src, mask=mask)
    return dst


def draw_roi_hist(src, mask=None):
    """
    绘制指定区域的直方图
    :param src:
    :param mask:
    :return:
    """
    if mask is None:
        hist = cv.calcHist([src], [0], None, [256], [0, 256])
    else:
        hist = cv.calcHist([src], [0], mask, [256], [0, 256])

    mask_img = extract_circular_regoin(src, mask)

    plt.subplot(221), plt.imshow(src, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(mask_img, 'gray')
    plt.subplot(224), plt.plot(hist)
    plt.xlim([0, 256])


def remove_line_near_edge(lines, circle):
    """
    移除边界线段
    :param lines: shape=(number of lines, 1, 4)
    :param circle:
    :return:
    """
    [center_x, center_y, ridius] = circle   # 获取圆的参数
    ridius_square = ridius * ridius

    dis_pt1 = np.square(lines[:, 0, 0] - center_x) + np.square(lines[:, 0, 1] - center_y)   # dis_pt1.shape=(22,)
    dis_pt2 = np.square(lines[:, 0, 2] - center_x) + np.square(lines[:, 0, 3] - center_y)   # dis_pt2.shape=(22,)
    dis_2_pt = dis_pt1 + dis_pt2
    # dis_2_pt.reshape(4, -1)

    dis_in = lines[dis_2_pt < (2 * 0.8 * ridius_square)]
    return dis_in


def detect_level_line(src, circle):
    """
    hough变换检测图像直线
    :param src:
    :return:
    """
    edges = cv.Canny(src, 50, 150, apertureSize=3)
    cv.imshow("edges", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=2)
    # print("直接检测到的直线", lines.shape)
    bgr = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    lines = remove_line_near_edge(lines, circle)    # lines.shape=(valid number of lines, 1, 4)
    # print("优化后检测到的直线", lines.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('line', bgr)

    return lines


def get_liquid_level_line(input_img):
    """
    获得液位线
    :param input_img:
    :return:
    """
    # 读取图像
    gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (3, 3), 0)     # 图像平滑


    # 找出图像中的圆形
    circles = find_circles(input_img)
    print(circles)
    # 在图像种绘制出找到的圆形
    input_img_copy = input_img.copy()
    for i in circles[0, :]:
        # 画圆
        cv.circle(input_img_copy, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # 画圆心
        cv.circle(input_img_copy, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.namedWindow('circles in example')
    cv.imshow('circles in example', input_img_copy)


    # 提高液位线与环境对比度
    # 没有直方图均衡化前的图像直方图
    # draw_roi_hist(blur_img, mask)
    # plt.show()

    equ_img = cv.equalizeHist(blur_img)
    cv.namedWindow("histogram equalization", cv.WINDOW_AUTOSIZE)
    cv.imshow("histogram equalization", equ_img)

    mask1 = get_circular_mask(input_img.shape[:2], [circles[0, 0, 0], circles[0, 0, 1], circles[0, 0, 2]])
    circle_img = extract_circular_regoin(blur_img, mask1)

    # 感兴趣区域经过直方图均衡化前的直方图
    # draw_roi_hist(blur_img, mask)
    # plt.show()


    # 检测液位线
    lines = detect_level_line(circle_img, circles[0, 0])
    # print(lines.shape)
    # 将获得的所有线段进行判断得到一条直线
    discrete_points = lines.reshape(-1, 2)
    zeros_img = np.zeros(blur_img.shape)
    for i in range(discrete_points.shape[0]):
        zeros_img[discrete_points[i, 1], discrete_points[i, 0]] = 255


    line_info = cv.fitLine(discrete_points, cv.DIST_FAIR, 0, 0.01, 0.01)
    # y = k * x + b
    line_k = line_info[1] / line_info[0]
    line_b = line_info[3] - line_k * line_info[2]

    pt1 = (0, np.int16(line_k * 0 + line_b))
    pt2 = (180, np.int16(line_k * 180 + line_b))
    cv.line(zeros_img, pt1, pt2, 255, 1)
    cv.imshow("zeros_img", zeros_img)

    # cv.waitKey(0)
    cv.destroyAllWindows()

    return line_k, line_b


if __name__ == '__main__':
    example = cv.imread('example/01.jpg')
    line_k, line_b = get_liquid_level_line(example)