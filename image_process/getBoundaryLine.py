import cv2 as cv
import numpy as np
from image_process.utils.utils import capture_roi, cal_largest_value, remove_small_contours


def cal_four_point_xy(src_img):
    """
    获取四个边界点
    :param source_img: 输入RGB图像
    :return: center_xy = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    # 显示原图像
    if __name__ == '__main__':
        cv.namedWindow("00_input", cv.WINDOW_KEEPRATIO)
        cv.imshow("00_input", src_img)

    # 获得液体住两侧区域
    roi_img = capture_roi(src_img, False)
    if __name__ == '__main__':
        cv.namedWindow("01_roi", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_roi", roi_img)
        cv.imwrite('example/side.png', roi_img)

    # 提取图中蓝色背景部分
    hsv_img = cv.cvtColor(roi_img, cv.COLOR_BGR2HSV)
    low_hsv = np.array([70, 6, 38])
    high_hsv = np.array([143, 255, 255])
    blue_img = cv.cv2.inRange(hsv_img, low_hsv, high_hsv)   # 灰度图像
    if __name__ == '__main__':
        cv.namedWindow("01_blue", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_blue", blue_img)

    # 腐蚀膨胀
    kernel = np.ones((3, 3), np.uint8)
    # kernel1 = np.ones((3, 3), np.uint8)
    mor_img = cv.dilate(blue_img, kernel, iterations=1)     # 膨胀
    # mor_img = cv.erode(mor_img, kernel, iterations=1)   # 腐蚀
    # fil_img = cv.morphologyEx(mask_img, cv.MORPH_CLOSE, kernel)
    if __name__ == '__main__':
        cv.namedWindow("02_0_morp", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_0_morp", mor_img)

    # 图像平滑
    # 防止下一步提取到的轮廓线断裂
    fil_img = cv.GaussianBlur(mor_img, (5, 5), 0)
    if __name__ == '__main__':
        cv.namedWindow("02_1_filter", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_1_filter", fil_img)

    # 提取轮廓
    edg_img = cv.Canny(fil_img, 80, 200, apertureSize=3)
    if __name__ == '__main__':
        cv.namedWindow("03_0_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_0_edges", edg_img)
        cv.imwrite('example/edge.png', edg_img)

    # 边界填充，padding = 4
    edg_img = cv.copyMakeBorder(edg_img, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # 去除小轮廓
    edg_img = remove_small_contours(edg_img, 8)
    if __name__ == '__main__':
        cv.namedWindow("03_2_edges", cv.WINDOW_KEEPRATIO)
        cv.imshow("03_2_edges", edg_img)

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

    # 边界删除，padding = 4
    blr_img = blr_img[2: -2, 2: -2]

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

    # 计算4(或者3， 2)个轮廓的质心
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
        for c_xy in center_xy:
            # temp_img[c_xy[i][1], center_xy[i][0]] = (0, 0, 255)
            cv.circle(temp_img, c_xy, radius=1, color=(0, 0, 255), thickness=2)

        # cv.line(temp_img, center_xy[0], center_xy[1], (0, 0, 255), 1, cv.LINE_AA)
        # cv.line(temp_img, center_xy[2], center_xy[3], (0, 0, 255), 1, cv.LINE_AA)
        cv.namedWindow("07_center", cv.WINDOW_KEEPRATIO)
        cv.imshow("07_center", temp_img)

    # 对标记线进行上下分类
    mark_points = classify_up_down(center_xy)

    return mark_points


def classify_up_down(center_xy):
    """
    根据中心点的y值，按从小到大对center_xy排序
    :param
    :return:
    """
    center_xy = np.array(center_xy)
    sort_index = np.argsort(center_xy[:, 1])
    center_xy = center_xy[sort_index]
    center_xy = center_xy.tolist()
    center_xy = [tuple(center_xy[i]) for i in range(len(center_xy))]    # center_xy:[( , ), ..., ( , )]

    length = len(center_xy)
    if length == 2:     # 有上下2个标记点
        symbol = 11
    elif length == 3:   # 有上下共3个标记点
        if center_xy[2][1] - center_xy[1][1] < 10:  # 比较两点的竖直方向的距离
            symbol = 12
        else:
            symbol = 21
    else:   # 有4个完整的标记点
        symbol = 22
    return {'symbol': symbol, 'points': center_xy}


if __name__ == '__main__':
    img = cv.imread('./example/gear00015.png')
    print(cal_four_point_xy(img))

    cv.waitKey()
    cv.destroyAllWindows()


