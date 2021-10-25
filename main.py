from image_process.getBoundaryLine import cal_four_point_xy
from image_process.getLiquidLevelLine import get_liquid_line
from PIL import Image
from image_detection.yolo import YOLO
import cv2 as cv
import numpy as np
import copy

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        # 齿轮箱液位孔定位
        r_image, top, bottom, left, right = yolo.detect_image(image)
        r_image.show()
        image = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        detect_image = image[top: bottom, left: right]
        cv.imwrite('image_process/example/gear' + img[-9:], detect_image)
        detect_image = cv.imread('image_process/example/gear' + img[-9:])

        # 获得液位线y = kx + b ==> -y + kx + b = 0
        detect_image_cp = copy.deepcopy(detect_image)
        detect_image_cp1 = copy.deepcopy(detect_image)
        line_k, line_b = get_liquid_line(detect_image)
        img_width = detect_image.shape[1]
        pt1 = (0, np.int16(line_k * 0 + line_b))
        pt2 = (img_width, np.int16(line_k * img_width + line_b))
        cv.line(detect_image, pt1, pt2, (0, 255, 0), 2)
        image[top: bottom, left: right] = detect_image
        cv.namedWindow("01_liquidLevel_img", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_liquidLevel_img", image)

        # 获得四个边界点
        # 限制边界线的红线在液位孔附近，可以直截取附近的图像来检测边界线
        center_xy = cal_four_point_xy(detect_image_cp1)
        print(center_xy)
        cv.line(detect_image, center_xy[0], center_xy[1], (0, 0, 255), 2, cv.LINE_AA)
        cv.line(detect_image, center_xy[2], center_xy[3], (0, 0, 255), 2, cv.LINE_AA)
        image[top: bottom, left: right] = detect_image
        cv.namedWindow("02_boundary_img", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_boundary_img", image)

        # 判断液位线是否正常
        value_top1 = -center_xy[0][1] + line_k * center_xy[0][0] + line_b
        value_top2 = -center_xy[1][1] + line_k * center_xy[1][0] + line_b
        value_bottom1 = -center_xy[2][1] + line_k * center_xy[2][0] + line_b
        value_bottom2 = -center_xy[3][1] + line_k * center_xy[3][0] + line_b
        # print(value_top1, value_top2, value_bottom1, value_bottom2)

        if (value_top1 * value_top2) > 0 and (value_bottom1 * value_bottom2 > 0) and (value_top1 * value_bottom1 < 0):
            print("液位线正常")
        else:
            print("液位线异常")

        cv.waitKey(0)
        cv.destroyAllWindows()


# ./image_detection/data/org_png/00011.png




