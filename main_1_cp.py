from border_detection.yolo import YOLO as LYOLO
from PIL import Image
from image_detection.yolo import YOLO
import cv2 as cv
import numpy as np
import copy
import os


def draw_line(detect_image, border_pt, llline_pt):
    """
    绘制标记线和液位线
    :param border_pt: 边界线,一般情况下array.shape = (4, 3)
    :param llline_pt: 液位线, array.shape =(4,)
    :return:
    """

    if border_pt.shape[0] == 4:     # 图像中画出3条线
        border_pt = border_pt.astype(np.int)
        detect_image = cv.cvtColor(np.asarray(detect_image), cv.COLOR_RGB2BGR)
        cv.line(detect_image, tuple([border_pt[0, 2], border_pt[0, 1]]), tuple([border_pt[1, 2], border_pt[1, 1]]), (0, 255, 0), 1)
        cv.line(detect_image, tuple([border_pt[2, 2], border_pt[2, 1]]), tuple([border_pt[3, 2], border_pt[3, 1]]), (0, 255, 0), 1)

        if llline_pt is not None:
            llline_pt = llline_pt.astype(np.int)
            cv.line(detect_image, tuple([llline_pt[1], llline_pt[0]]), tuple([llline_pt[3], llline_pt[2]]), (0, 0, 255), 1)
        cv.namedWindow('line', cv.WINDOW_KEEPRATIO)
        cv.imshow('line', detect_image)
    elif border_pt.shape[0] == 3:    # 图像中画出1边界线，1液位线，1个单独点
        border_pt = border_pt.astype(np.int)
        detect_image = cv.cvtColor(np.asarray(detect_image), cv.COLOR_RGB2BGR)
        if border_pt[0, 0] == 0 & border_pt[1, 0] == 0:     # 绘制上边界线和下方的孤立点
            cv.line(detect_image, tuple([border_pt[0, 2], border_pt[0, 1]]), tuple([border_pt[1, 2], border_pt[1, 1]]),
                (0, 255, 0), 1)
            cv.circle(detect_image, (border_pt[2, 2], border_pt[2, 1]), radius=1, color=(0, 255, 0), thickness=0)
        else:   # 绘制下边界线和上方的孤立点
            cv.line(detect_image, tuple([border_pt[1, 2], border_pt[1, 1]]), tuple([border_pt[2, 2], border_pt[2, 1]]),
                (0, 255, 0), 1)
            cv.circle(detect_image, (border_pt[0, 2], border_pt[0, 1]), radius=1, color=(0, 255, 0), thickness=1)

        if llline_pt is not None:
            llline_pt = llline_pt.astype(np.int)
            cv.line(detect_image, tuple([llline_pt[1], llline_pt[0]]), tuple([llline_pt[3], llline_pt[2]]), (0, 0, 255),
                    1)
        cv.namedWindow('line', cv.WINDOW_KEEPRATIO)
        cv.imshow('line', detect_image)
    else:
        pass


def get_line_kbmodel(line):
    """
    直线的方程,自变量为像素距图像上边沿的距离，因变量为像素距图像做边沿的距离
    :param line= [y1, x1, y2, x2], [因，自，因，自]
    :return: line_k, line_b
    """
    if line[1] == line[3]:
        line[1] += 1
    line_k = (line[2] - line[0]) / (line[3] - line[1])
    line_b = -line_k * line[1] + line[0]

    return line_k, line_b


def llline_status(pos_cls_info):
    """

    :param pos_cls_info:
    :return:
    """
    Flag = 1    # 0——异常，1——正常， -1——程序没有检测到液位柱
    pos_cls_info = np.array(pos_cls_info, dtype=np.float).reshape(-1, 5)

    # 获取3条线的端点,并进行数据处理
    border = pos_cls_info[pos_cls_info[:, -1] == 0]  # 获得2条标记线的信息
    llline = pos_cls_info[pos_cls_info[:, -1] == 1]  # 获得液位线的信息
    if 4 >= border.shape[0] >= 3:
        if llline.size == 0:
            llline = np.array([0, 0, 0, 0])
            Flag = 0
            return Flag
    # elif pos_cls_info.shape[0] > 4: # 检测的标记线多了，有问题
    else:
        Flag = -1

    # 获得边界线的端点
    border_pt = np.zeros((border.shape[0], 2), dtype=float)  # 一般情况下，shape = (4, 2)
    border_pt[:, 0] = (border[:, 0] + border[:, 2]) / 2  # 获得边界线bbox的中心点
    border_pt[:, 1] = (border[:, 1] + border[:, 3]) / 2
    border_pt = border_pt[np.argsort(border_pt[:, 0])]  # 点按照y值从小到大排序

    # 当图像中检测到液位线时
    if np.any(llline):
        # 获得液位线的中点
        llline = llline.reshape(-1)
        llline_c_pt = np.zeros((2,), dtype=np.float)
        llline_c_pt[0] = (llline[0] + llline[2]) / 2  # 获得边界线bbox的中心点
        llline_c_pt[1] = (llline[1] + llline[3]) / 2

        # 判断液位线是否正常 & 获得边界线的数学模型
        if border_pt.shape[0] == 4:     # 获得4条标记线
            line_k, line_b = get_line_kbmodel(border_pt[:2].reshape(-1))    # 获得上方边界线的数学模型
            if (border_pt[1, 0] < llline_c_pt[0] < border_pt[2, 0]):
                Flag = 1
            else:
                Flag = 0
            border_pt = np.insert(border_pt, 0, [0, 0, 1, 1], axis=1)

        elif border_pt.shape[0] == 3:     # 获得3条标记线
            if abs(border_pt[0, 0] - border_pt[1, 0]) < abs(border_pt[0, 0] - border_pt[2, 0]):   # 3个点中的2个点在上方
                line_k, line_b = get_line_kbmodel(border_pt[:2].reshape(-1))    # 获得上方边界线的数学模型
                if border_pt[1, 0] < llline_c_pt[0] < border_pt[2, 0]:
                    Flag = 1
                else:
                    Flag = 0
                border_pt = np.insert(border_pt, 0, [0, 0, 1], axis=1)
            else:   # 3个点中的2个点在下方
                line_k, line_b = get_line_kbmodel(border_pt[1: 3].reshape(-1))  # 获得下方边界线的数学模型
                if border_pt[0, 0] < llline_c_pt[0] < border_pt[1, 0]:
                    Flag = 1
                else:
                    Flag = 0
                border_pt = np.insert(border_pt, 0, [0, 1, 1], axis=1)
        else:
            Flag = -1

        # 根据已知的边界线数学模型， 来绘制液位线
        llline_pt = np.zeros((4,), dtype=np.float)
        llline_pt[1] = llline[1]
        llline_pt[3] = llline[3]
        # 计算边界线线与液位线的距离
        distance = llline_c_pt[0] - (line_k * llline_c_pt[1] + line_b)
        llline_pt[0] = line_k * llline[1] + line_b + distance
        llline_pt[2] = line_k * llline[3] + line_b + distance

        # 绘制液位线
        print(border_pt)
        print("line", line_k, " ", line_b)
        draw_line(detect_image, border_pt, llline_pt)

    else:
        # 绘制液位线
        draw_line(detect_image, border_pt, None)

    return Flag


yolo = YOLO()
lyolo = LYOLO()


while(True):
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        # 齿轮箱液位孔定位
        r_image, top, bottom, left, right = yolo.detect_image(image)
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            continue
        r_image.show()
        image = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        detect_image = image[top: bottom, left: right]
        cv.imwrite('image_process/example/gear' + "_temp.png", detect_image)

        # 获得液位线y = kx + b ==> -y + kx + b = 0
        detect_image = Image.open('image_process/example/gear' + '_temp.png')
        _, pos_cls_info = lyolo.detect_image(detect_image)
        print(pos_cls_info)

        # 数据处理
        flag = llline_status(pos_cls_info)
        print(flag)

        # 判断液位线是否正常
        # Flag = line_is_normal([line_k, line_b], mark_points)
        # # print(value_top1, value_top2, value_bottom1, value_bottom2)
        #
        # if Flag:
        #     print("液位线正常")
        # else:
        #     print("液位线异常")

        cv.waitKey()
        cv.destroyAllWindows()


# image_detection/data/org_png/00056.png
# data/202207/40.png