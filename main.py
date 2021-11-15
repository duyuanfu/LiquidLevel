from image_process.getBoundaryLine import cal_four_point_xy
from image_process.getLiquidLevelLine import get_liquid_line
from PIL import Image
from image_detection.yolo import YOLO
import cv2 as cv
import numpy as np
import copy
import os


def line_is_normal(line_kb, mark_points):
    [line_k, line_b] = line_kb
    symbol = mark_points['symbol']
    points = np.array(mark_points['points'])
    value = -points[:, 1] + line_k * points[:, 0] + line_b
    # print(points, line_k, line_b)
    # print(value)

    # 当液位线低于下方标记线，液位线检测程序将设置line_k == 0，line_b == 0
    if line_k == 0 and line_b == 0:
        return False

    if symbol == 22:
        if (value[0] * value[1] > 0) and (value[2] * value[3] > 0) and (value[0] * value[2] < 0):
            return True
        else:
            return False
    elif symbol == 12:
        if (value[0] * value[1] < 0) and (value[1] * value[2] > 0):
            return True
        else:
            return False
    elif symbol == 21:
        if (value[0] * value[1] > 0) and (value[0] * value[2] < 0):
            return True
        else:
            return False
    else:
        if value[0] * value[1] < 0:
            return True
        else:
            return False


yolo = YOLO()

img_path = 'E:/Pycharm/Workplace/LiquidLevel/image_detection/data/org_png/'  # 读取图像文件夹
for file in os.listdir(img_path):   # 遍历访问图像
    img = img_path + file
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
        print('image:' + file + '\'gear box:%d, %d,%d, %d,' % (top, bottom, left, right))
        r_image.show()
        image = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        detect_image = image[top: bottom, left: right]
        cv.imwrite('image_process/example/gear' + img[-9:], detect_image)
        detect_image = cv.imread('image_process/example/gear' + img[-9:])

        # 获得液位线y = kx + b ==> -y + kx + b = 0
        detect_image_cp = copy.deepcopy(detect_image)
        line_k, line_b = get_liquid_line(detect_image)
        if line_k != 0 or line_b != 0:
            img_width = detect_image.shape[1]
            pt1 = (0, np.int16(line_k * 0 + line_b))
            pt2 = (img_width, np.int16(line_k * img_width + line_b))
            cv.line(detect_image, pt1, pt2, (0, 255, 0), 2)
            image[top: bottom, left: right] = detect_image
            # cv.namedWindow("01_liquidLevel_img", cv.WINDOW_KEEPRATIO)
            # cv.imshow("01_liquidLevel_img", image)

        # 获得四个边界点
        mark_points = cal_four_point_xy(detect_image_cp)
        # cv.line(detect_image, center_xy[0], center_xy[1], (0, 0, 255), 2, cv.LINE_AA)
        # cv.line(detect_image, center_xy[2], center_xy[3], (0, 0, 255), 2, cv.LINE_AA)
        points = mark_points['points']
        for point in points:
            cv.circle(detect_image, point, radius=1, color=(0, 0, 255), thickness=2)
        image[top: bottom, left: right] = detect_image
        cv.namedWindow("02_boundary_img", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_boundary_img", image)

        # 判断液位线是否正常
        Flag = line_is_normal([line_k, line_b], mark_points)
        # print(value_top1, value_top2, value_bottom1, value_bottom2)

        if Flag:
            print("液位线正常")
        else:
            print("液位线异常")

        cv.waitKey(1000)
        cv.destroyAllWindows()
