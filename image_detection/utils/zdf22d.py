import argparse
from pathlib import Path
import numpy as np
import cv2
import zivid
import os


if __name__ == '__main__':
    src_path = 'G:/TrainInspection/picture/yewei/'  # 读取图像文件夹
    dst_path = '../data/org_png/'   # 保存图像文件夹
    for file in os.listdir(src_path):   # 遍历访问图像
        filename = src_path + file

        app = zivid.Application()
        frame = zivid.Frame(filename)
        rgba = frame.point_cloud().copy_data("rgba")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        save_path = (dst_path + file.strip('zdf') + 'png')
        print(save_path)
        cv2.imwrite(save_path, bgr)

