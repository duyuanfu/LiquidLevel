import cv2 as cv
import numpy as np
import os


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


def get_line_kbmodel(line):
    """
    :param line= [w1, h1, w2, h2], 线的方程自变量为像素距左上角的高度，因变量为像素距左上角的宽度
    :return: line_k, line_b
    """
    if line[1] == line[3]:
        line[1] += 1
    line_k = (line[2] - line[0]) / (line[3] - line[1])
    line_b = -line_k * line[1] + line[0]

    return line_k, line_b


def is_close_line(lines):
    """

    :param lines: [[x1, y1, x2, y2],[ , , , ]]
    :return:
    """
    line1 = lines[0]
    line2 = lines[1]

    # 第一种情况：两条线的端点比较接近
    point_thresold = 100  # 25是一个阈值，手动设定
    if np.square(line1[0] - line2[0]) + np.square(line1[1] - line2[1]) < point_thresold:
        return True
    if np.square(line1[0] - line2[2]) + np.square(line1[1] - line2[3]) < point_thresold:
        return True
    if np.square(line1[2] - line2[2]) + np.square(line1[3] - line2[3]) < point_thresold:
        return True
    if np.square(line1[2] - line2[0]) + np.square(line1[3] - line2[1]) < point_thresold:
        return True

    # 第二种情况：两直线比较平行，有一部分靠近近似重叠
    # 有点难，先放着

    return False


def is_close_all_lines(lines):
    """
    两两判断线段是否接近，若接近删除其中长度较短的一条
    :param lines: np.array[[x1, y1, x2, y2],[ , , , ], ..., [, , , ]]
    :return:lines——优化后的两条直线。
    """
    num_lines = lines.shape[0]

    # 计算线段的长度
    len_lines = np.square(lines[:, 0] - lines[:, 2]) + np.square(lines[:, 1] - lines[:, 3])
    len_lines = np.argsort(-len_lines)  # 根据线段的长度进行降序排序
    lines = lines[len_lines]

    for i in range(0, num_lines - 1):
        for j in range(i + 1, num_lines):
            if is_close_line([lines[i], lines[j]]):  # 若两直线接近，使长度较短者为0
                lines[j] = [0, 0, 0, 0]
                continue

    zero_row_idx = np.argwhere(np.all(lines[:, ...] == 0, axis=1))
    lines = np.delete(lines, zero_row_idx, axis=0)

    # 当剩余的线段还大于2时，通过判断长度返回最长的两条线段

    return lines[:2]


def adapt_houghlinesP(edg_img):
    """
    对图片进行HoughLinesP可能会得到多条线段，而实际我们只需要两条竖线，所以要对获取到的线段进
    行优化
    :return:
    """
    threshold = 21
    minLineLength = 40
    maxLineGap = 20

    while True:
        lines = cv.HoughLinesP(edg_img, rho=1, theta=np.pi / 180, threshold=threshold, minLineLength=minLineLength,
                               maxLineGap=maxLineGap)
        if lines is None:
            threshold = threshold - 5
            threshold = (43 if threshold <= 0 else threshold)   # 解决cv.HoughLinesP的小bug
            continue
        lines = np.array(lines).reshape(-1, 4)
        # 线的斜率若大于一定值，则舍去
        for i in range(lines.shape[0]):
            k, _ = get_line_kbmodel(lines[i])
            if k > 0.7:
                lines[i] = np.zeros(lines.shape[1])
        lines = lines[[not np.all(lines[i] == 0) for i in range(lines.shape[0])], :]

        num_lines = lines.shape[0]
        if num_lines < 2:  # 检测到的线段数量小于2时， 降低阈值
            threshold = threshold - 5
        elif num_lines == 2:  # 判断这2条线段是否接近
            if not is_close_line(lines):
                break
            else:
                threshold = threshold - 5
        elif num_lines > 6:     # 检测到的线段太多，增加阈值
            threshold = threshold + 5
        else:   # 当线段数量大于2时, 优化只保留2条直线
            lines = is_close_all_lines(lines)
            break
    return lines


def get_roi(input_img, pt_list, inner):
    """
    提取两条直线之间的区域，线的方程自变量为像素距左上角的高度，因变量为像素距左上角的宽度
    :param input_img: rgb图像
    :param pt_list: 两条直线4个点的坐标
    :param inner: True——取两直线之间的区域，False——取两直线之外的区域
    :return:
    """
    pt1 = pt_list[0]
    pt2 = pt_list[1]
    # print(pt1, pt2)

    k1, b1 = get_line_kbmodel(pt1)
    k2, b2 = get_line_kbmodel(pt2)
    # print(k1, b1, k2, b2)

    roi_img = np.zeros(input_img.shape, dtype=input_img.dtype)
    if inner:   # 取两直线之间的区域
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                x1 = k1 * i + b1
                x2 = k2 * i + b2
                if x1 >= j >= x2 or x1 <= j <= x2:
                    roi_img[i, j] = input_img[i, j]
    else:   # 取两直线之外的区域
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                x1 = k1 * i + b1
                x2 = k2 * i + b2
                if not (x1 >= j >= x2 or x1 <= j <= x2):
                    roi_img[i, j] = input_img[i, j]

    return roi_img


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


def remove_small_contours(input_img, num_keep):
    """
    :param input_img: 灰度图像
    :param num_keep: 保留轮廓的数目
    :return: 返回保留了较大轮廓的图像
    """
    cnt_img, contours, hierarchy = cv.findContours(input_img, 1, 2)
    if __name__ == '__main__':
        temp_img = cv.cvtColor(input_img, cv.COLOR_GRAY2BGR)
        # rect_area = []  # 记录轮廓最小外接矩阵的面积
        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(temp_img, [box], 0, (0, 0, 255), 1)
            # rect_area.append(rect[1][0] * rect[1][1])

        cv.namedWindow("03_1_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_1_edges", temp_img)

    # 记录轮廓最小外接矩阵的面积
    rect_area = [(cv.minAreaRect(contours[i])[1][0] * cv.minAreaRect(contours[i])[1][1]) for i in
                 range(len(contours))]
    max_area_index = cal_largest_value(rect_area, num_keep)  # 获取轮廓最小外接矩阵的面积最大几部分的轮廓
    for i in range(len(rect_area)):
        if i not in max_area_index:  # 将其余轮廓的灰度值置0
            cv.drawContours(cnt_img, [contours[i]], 0, 0, -1)

    return cnt_img


def remove_contours_thresh(input_img, thresh):
    """
    :param input_img: 灰度图像
    :param num_keep: 保留轮廓的数目
    :return: 返回保留了较大轮廓的图像
    """
    cnt_img, contours, hierarchy = cv.findContours(input_img, 1, 2)
    if __name__ == '__main__':
        temp_img = cv.cvtColor(input_img, cv.COLOR_GRAY2BGR)
        # rect_area = []  # 记录轮廓最小外接矩阵的面积
        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(temp_img, [box], 0, (0, 0, 255), 1)
            # rect_area.append(rect[1][0] * rect[1][1])

        cv.namedWindow("03_1_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_1_edges", temp_img)

    # 记录轮廓最小外接矩阵的面积
    rect_area = [(cv.minAreaRect(contours[i])[1][0] * cv.minAreaRect(contours[i])[1][1]) for i in
                 range(len(contours))]
    rect_area = np.array(rect_area)
    max_area_index = np.where(rect_area > thresh)[0]

    for i in range(len(rect_area)):
        if i not in max_area_index:  # 将其余轮廓的灰度值置0
            cv.drawContours(cnt_img, [contours[i]], 0, 0, -1)

    return cnt_img


def capture_roi(src_img, inner):
    """
    :param ext_img: RGB图像
    :return: roi_img——感兴趣区域
    """
    #  显示图像
    if __name__ == '__main__':
        cv.namedWindow("00_src", cv.WINDOW_KEEPRATIO)
        cv.imshow("00_src", src_img)

    hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    low_hsv = np.array([44, 0, 38])
    high_hsv = np.array([143, 255, 255])
    blue_img = cv.inRange(hsv_img, low_hsv, high_hsv)   # 灰度图像
    if __name__ == '__main__':
        cv.namedWindow("01_blue", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_blue", blue_img)

    # 腐蚀膨胀
    kernel = np.ones((7, 7), np.uint8)
    mor_img = cv.dilate(blue_img, kernel, iterations=1)  # 膨胀
    mor_img = cv.erode(mor_img, kernel, iterations=1)  # 腐蚀
    # fil_img = cv.morphologyEx(mask_img, cv.MORPH_CLOSE, kernel)
    if __name__ == '__main__':
        cv.namedWindow("02_morp", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_morp", mor_img)

    # 去除小轮廓
    edg_img = remove_small_contours(mor_img, 2)
    if __name__ == '__main__':
        cv.namedWindow("03_2_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_2_edges", edg_img)

    # 提取轮廓
    edg_img = cv.Canny(mor_img, 80, 200, apertureSize=3)
    if __name__ == '__main__':
        cv.namedWindow("03_3_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_3_edges", edg_img)

    # 获取液体住两侧的竖直的边界线
    lines = adapt_houghlinesP(edg_img)
    if __name__ == '__main__':
        temp_img = src_img
        for i in range(lines.shape[0]):
            x1, y1, x2, y2 = lines[i]
            # cv.line(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.namedWindow('04_line', cv.WINDOW_KEEPRATIO)
            cv.imshow('04_line', temp_img)

    # 取出液位柱两侧区域
    roi_img = get_roi(src_img, lines, inner)
    if __name__ == '__main__':
        cv.namedWindow('05_roi', cv.WINDOW_KEEPRATIO)
        cv.imshow('05_roi', roi_img)

    return roi_img


def get_top_edge(input_img):
    """

    :param input_img: 二值化图像
    :return: top_edge = [(x1, y1), (x2, y2), ...(xn, yn)]上边沿所在点组成的列表
    """
    height, width = input_img.shape
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


def extract_level_color(src_img):
    # 提取黄色区域
    hsv = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    low_hsv = np.array([0, 90, 46])     # 60 -> 90 ->
    high_hsv = np.array([40, 255, 255])
    yel_img = cv.inRange(hsv, low_hsv, high_hsv)   # 灰度图像
    if __name__ == '__main__':
        cv.namedWindow("03_mask", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_mask", yel_img)

    # 提取上边沿
    top_edge = get_top_edge(yel_img)
    top_edge = np.array(top_edge).reshape(-1, 2)
    # 当液位线低于下方标记线时，返回line_k = 0, line_b = 0
    if top_edge.shape[0] <= 25:
        return 0, 0

    # 将构成上边沿的离散点拟合成一条直线
    line_info = cv.fitLine(top_edge, cv.DIST_HUBER, 0, 0.01, 0.01)
    # y = k * x + b
    line_k = line_info[1] / line_info[0]
    line_b = line_info[3] - line_k * line_info[2]

    return line_k, line_b


def extract_level_sobel(src_img):
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    if __name__ == '__main__':
        cv.namedWindow('00_src_img', cv.WINDOW_KEEPRATIO)
        cv.imshow('00_src_img', gray_img)

    # 剔除两旁的黑色区域
    img_height, img_width = gray_img.shape
    x_topleft, x_bottomleft, x_topright, x_bottomright = 0, 0, img_width, img_width

    for i in range(img_width):
        if gray_img[0, i] != 0:
            x_topleft = i
            break
    for i in range(img_width):
        if gray_img[img_height - 1, i] != 0:
            x_bottomleft = i
            break
    for i in range(img_width, 0, -1):
        if gray_img[0, i - 1] != 0:
            x_topright = i
            break
    for i in range(img_width, 0, -1):
        if gray_img[img_height - 1, i - 1] != 0:
            x_bottomright = i
            break

    x_left = x_topleft if x_topleft > x_bottomleft else x_bottomleft
    x_right = x_topright if x_topright < x_bottomright else x_bottomright
    x_width = int(x_right - x_left)

    x_left = x_left + int(x_width * 0.15)
    x_right = x_right - int(x_width * 0.15)

    inner_img = gray_img[:, x_left: x_right]
    inn_height, inn_width = inner_img.shape
    if __name__ == '__main__':
        cv.namedWindow('01_inn_img', cv.WINDOW_KEEPRATIO)
        cv.imshow('01_inn_img', inner_img)

    # sobel滤波
    sobel_img = cv.Sobel(inner_img, cv.CV_8U, 0, 1, ksize=3, scale=1)
    sobel_img = cv.convertScaleAbs(sobel_img)  # 转回uint8
    if __name__ == '__main__':
        cv.namedWindow('02_sobel_img', cv.WINDOW_KEEPRATIO)
        cv.imshow('02_sobel_img', sobel_img)

    # 标准化
    blockSize = min(inn_height, inn_width)
    blockSize = blockSize - 1 if blockSize % 2 == 0 else blockSize
    mean_gray, std_gray = cv.meanStdDev(sobel_img)
    print(mean_gray, std_gray)
    nor_img = ((sobel_img - mean_gray) / (std_gray * std_gray) + 1) * 122.5
    nor_img = cv.convertScaleAbs(nor_img)
    if __name__ == '__main__':
        cv.namedWindow('03_nor_img', cv.WINDOW_KEEPRATIO)
        cv.imshow('03_nor_img', nor_img)

    # 二值化
    # bin_img = cv.adaptiveThreshold(nor_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=blockSize,
    #                                C=-4)
    ret, bin_img = cv.threshold(sobel_img, 50, 255, cv.THRESH_BINARY)
    if __name__ == '__main__':
        cv.namedWindow('04_bin_img', cv.WINDOW_KEEPRATIO)
        cv.imshow('04_bin_img', bin_img)

    # 去除离散点
    # cnt_img = remove_small_contours(bin_img, 1)
    cnt_img = remove_contours_thresh(bin_img, 20)

    # 获取灰度值为255的点的坐标
    line_points = []
    for i in range(inn_height):
        for j in range(inn_width):
            if cnt_img[i, j] != 0:
                line_points.append([i, j])

    line_points = np.array(line_points).reshape(-1, 2)
    if line_points.shape[0] < 20:   # 液位线不在roi图像中
        return 0, 0

    # y方向上压缩line_points,
    compress_rate = 0.5     # 压缩率
    average_y = np.average(line_points[:, 0])
    line_points[:, 0] = average_y + compress_rate * (line_points[:, 0] - average_y)

    line_points[:, [0, 1]] = line_points[:, [1, 0]]
    line_info = cv.fitLine(line_points, cv.DIST_L1, 0, 0.01, 0.01)  # 第2个参数也可以是cv.DIST_FAIR
    line_k = line_info[1] / line_info[0]
    # - line_k * x_left ，获得原图中的直线坐标
    line_b = line_info[3] - line_k * line_info[2] - line_k * x_left
    # print(line_k, line_b, line_k * x_left)

    # 绘制液位线
    if __name__ == '__main__':
        # 在裁剪区域上绘制
        line_b = line_info[3] - line_k * line_info[2]
        pt1 = (0, np.int16(line_k * 0 + line_b))
        pt2 = (inn_width, np.int16(line_k * inn_width + line_b))
        bin_img = cv.cvtColor(bin_img, cv.COLOR_GRAY2BGR)
        cv.line(bin_img, pt1, pt2, (255, 0, 0), 1)
        cv.namedWindow("05_line1", cv.WINDOW_KEEPRATIO)
        cv.imshow("05_line1", bin_img)

        # 在原图上绘制
        line_b = line_info[3] - line_k * line_info[2] - line_k * x_left
        pt1 = (0, np.int16(line_k * 0 + line_b))
        pt2 = (img_width, np.int16(line_k * img_width + line_b))
        cv.line(src_img, pt1, pt2, (255, 0, 0), 1)
        cv.namedWindow("05_line", cv.WINDOW_KEEPRATIO)
        cv.imshow("05_line", src_img)

    return line_k, line_b


def get_all_roi():
    path_img = '../example/'
    i = 0
    for filename in os.listdir(path_img):
        if filename.find('gear') != -1:
            img = cv. imread(os.path.join(path_img, filename))
            roi_img = capture_roi(img, inner=True)
            print(os.path.join(path_img, str('roi%05d' % i)))
            cv.imwrite(os.path.join(path_img, str('roi%05d' % i) + '.png'), roi_img)
            i = i + 1


if __name__ == '__main__':

    # path_img = '../example/'
    # i = 0
    # for filename in os.listdir(path_img):
    #     if filename.find('roi') != -1:
    #         img = cv. imread(os.path.join(path_img, filename))
    #         extract_level_sobel(img)
    #         print(filename)
    #         cv.waitKey(1000)

    img = cv.imread('../example/roi00059.png')
    extract_level_sobel(img)

    cv.waitKey()
    cv.destroyAllWindows()
