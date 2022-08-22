import cv2 as cv
import numpy as np
from image_process.utils.utils import capture_roi, extract_level_sobel, extract_level_color


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

    # 获得液体柱区域
    roi_img = capture_roi(src_img, True)
    if __name__ == '__main__':
        cv.namedWindow("01_roi", cv.WINDOW_KEEPRATIO)
        cv.imshow("01_roi", roi_img)

    # 获取液位线
    line_k, line_b = extract_level_sobel(roi_img)
    # line_k, line_b = extract_level_color(roi_img)

    # 当液位低于标记线时，提取到的上边沿的离散点，导致斜率太大，需舍去
    if line_k < -0.7 or line_k > 0.7:
        return 0, 0

    if __name__ == '__main__':
        img_width = src_img.shape[1]
        pt1 = (0, np.int16(line_k * 0 + line_b))
        pt2 = (img_width, np.int16(line_k * img_width + line_b))
        cv.line(src_img, pt1, pt2, 255, 1)
        cv.namedWindow("02_line", cv.WINDOW_KEEPRATIO)
        cv.imshow("02_line", src_img)

    return line_k, line_b


if __name__ == '__main__':
    img = cv.imread('example/gear00004.png')
    print(get_liquid_line(img))

    cv.waitKey()
    cv.destroyAllWindows()
