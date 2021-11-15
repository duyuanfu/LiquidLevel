import cv2

# 定义窗口名称
winName = 'Region of specified color'


# 定义滑动条回调函数，此处pass用作占位语句保持程序结构的完整性
def nothing(x):
    pass


img_original = cv2.imread('../example/gear00059.png')
# 颜色空间的转换
img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
# 新建窗口
cv2.namedWindow(winName, cv2.WINDOW_GUI_NORMAL)
# 新建6个滑动条，表示颜色范围的上下边界，这里滑动条的初始化位置即为黄色的颜色范围
# 提取蓝色环境的6个值：44, 143, 0, 255, 38, 255
# 提取黄色液体的6个值：0, 40, 90, 255, 46, 255
# 提取液体柱上方空气柱的6个值：0, 50, 0, 150, 70, 255。q
# 提取红色标记线的6个值：0, 16, 50, 255，137, 255
cv2.createTrackbar('Hmin', winName, 0, 255, nothing)
cv2.createTrackbar('Hmax', winName, 40, 255, nothing)
cv2.createTrackbar('Smin', winName, 90, 255, nothing)
cv2.createTrackbar('Smax', winName, 255, 255, nothing)
cv2.createTrackbar('Vmin', winName, 46, 255, nothing)
cv2.createTrackbar('Vmax', winName, 255, 255, nothing)
while (1):
    # 函数cv2.getTrackbarPos()范围当前滑块对应的值
    Hmin = cv2.getTrackbarPos('Hmin', winName)
    Smin = cv2.getTrackbarPos('Smin', winName)
    Vmin = cv2.getTrackbarPos('Vmin', winName)
    Hmax = cv2.getTrackbarPos('Hmax', winName)
    Smax = cv2.getTrackbarPos('Smax', winName)
    Vmax = cv2.getTrackbarPos('Vmax', winName)
    # 得到目标颜色的二值图像，用作cv.bitwise_and()的掩模
    img_target = cv2.inRange(img_hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
    # 输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像
    img_specifiedColor = cv2.bitwise_and(img_original, img_original, mask=img_target)
    cv2.imshow(winName, img_specifiedColor)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
