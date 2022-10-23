# -*- coding:utf-8 -*-
"""
@author: Felix Z
"""

import cv2
import random
import numpy as np
from math import *
from PIL import Image


def generate_kernel(img_height, img_width, radius, center_x, center_y):
    """
    :param img_height: 掩码高度
    :param img_width: 掩码宽度
    :param radius: 掩码半径
    :param center_x: 掩码中心x坐标
    :param center_y: 掩码中心y坐标
    :return: 掩码
    """
    y, x = np.ogrid[0:img_height, 0:img_width]
    # circle mask
    kernel = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    return kernel


def edge_detection(inputs, radius):
    """
    :param inputs:
    :param radius: USAN卷积核的半径 => n+0.5
    :return: 处理后图像 (Image类)
    """
    global t, r, md

    oimg = Image.fromarray(inputs)
    img2 = oimg.convert('L')
    pixels2 = img2.load()

    kernel = generate_kernel(radius * 2, radius * 2, radius, int(radius), int(radius))
    print('kernel:')
    print(kernel)

    # m = [
    #     [0, 0, 1, 1, 1, 0, 0],
    #     [0, 1, 1, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [0, 1, 1, 1, 1, 1, 0],
    #     [0, 0, 1, 1, 1, 0, 0]
    # ]

    # m = [
    #     [0, 0, 1, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 1, 0, 0]
    # ]

    count = 37
    g = 3.0 * count / 4.0

    n = [[0 for i in range(img2.size[1])] for j in range(img2.size[0])]

    mx = [[0 for i in range(img2.size[1])] for j in range(img2.size[0])]
    my = [[0 for i in range(img2.size[1])] for j in range(img2.size[0])]

    total = img2.size[0] * img2.size[1]

    # Stage 1
    progress = 0
    for x in range(img2.size[0]):
        for y in range(img2.size[1]):
            progress += 1
            if progress % 10000 == 0:
                print('Progress:', progress, '/', total)
            for xr in range(md):
                for yr in range(md):
                    xx = x - r + xr
                    yy = y - r + yr
                    if kernel[xr][yr] == 1 and xx >= 0 and xx < img2.size[0] and yy >= 0 and yy < img2.size[1]:
                        cdif = c2(pixels2[xx, yy], pixels2[x, y])
                        n[x][y] += cdif
                        mx[x][y] += cdif * (xr - r)
                        my[x][y] += cdif * (yr - r)

    # Stage 2: 求取x和y方向边缘信息平均值
    print('正在求取x和y方向边缘信息平均值...')
    for x in range(img2.size[0]):
        for y in range(img2.size[1]):
            mx[x][y] /= n[x][y]
            my[x][y] /= n[x][y]

    for x in range(img2.size[0]):
        for y in range(img2.size[1]):
            if n[x][y] < g:
                n[x][y] = g - n[x][y]
            else:
                n[x][y] = 0

    root2 = sqrt(2)
    delta = 0.00000000000001

    # Stage 3
    print('-' * 200)
    progress = 0
    for x in range(img2.size[0]):
        for y in range(img2.size[1]):
            progress += 1
            if progress % 100000 == 0:
                print('Progress:', progress, '/', total)
            if mx[x][y] and my[x][y]:
                ang = atan2(my[x][y], mx[x][y])
                x1 = int(x + ceil(cos(ang) * root2 - delta))
                y1 = int(y + ceil(sin(ang) * root2 - delta))
                x2 = int(x - ceil(cos(ang) * root2 - delta))
                y2 = int(y - ceil(sin(ang) * root2 - delta))
                if 0 <= x1 < img2.size[0] and 0 <= y1 < img2.size[1] and 0 <= x2 < img2.size[
                    0] and 0 <= y2 < img2.size[1]:
                    if n[x][y] >= n[x1][y1] and n[x][y] >= n[x2][y2]:
                        pixels2[x, y] = int(n[x][y] * 255 / g)
                    else:
                        pixels2[x, y] = 0
            else:
                pixels2[x, y] = n[x][y] * 255 / g
    img2 = np.array(img2)
    return img2


def c(r, r0):
    global t
    if (abs(r - r0) > t):
        return 0
    else:
        return 1


def c2(r, r0):
    global t
    return exp(-((r - r0) / float(t)) ** 2)


def detect_edges(dir):
    oimg = cv2.imread(dir)
    # 降低分辨率
    factor = 0.2
    width = int(oimg.shape[0] * factor)
    length = int(oimg.shape[1] * factor)
    print('lowered pixels:', width, '*', length)
    oimg = cv2.resize(oimg, (length, width), interpolation=cv2.INTER_CUBIC)
    # print(type(oimg))
    # print(oimg.shape)
    img = edge_detection(oimg, r)
    print(img.shape)

    # 高斯滤波器
    # img = cv2.GaussianBlur(img, (3, 3), 1.3)

    # 显示图像
    cv2.imshow('GaussianBlur', img)
    cv2.waitKey(0)
    cv2.imwrite('test.jpg', img)

    # 保存图像
    # cv2.imwrite('0a0ae8699f1a5a.jpg', img)


def randomize_color(grey_img):
    """
    grey img must be np.array with shape (h, w) with grey scale
    """
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    rgb_img = np.array([[[0 for i in range(3)] for j in range(grey_img.shape[1])] for k in range(grey_img.shape[0])])
    print('Coloring:', rgb_img.shape, 'with color code:', (r, g, b))
    for x in range(grey_img.shape[0]):
        for y in range(grey_img.shape[1]):
            if grey_img[x][y] > 50:
                rgb_img[x, y] = (r, g, b)
    rgb_img = np.array(rgb_img, dtype=np.uint8)
    cv2.imshow('rgb', rgb_img)
    # cv2.waitKey(0)
    return rgb_img


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def transform(viddir):

    cap = cv2.VideoCapture(viddir)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)

    video = set_saved_video(cap, 'converted.mp4', (width, height))
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        print('currently on frame:', frame_num)
        if not ret:
            break
        if frame_num % per_n_frame == 0:
            img = randomize_color(edge_detection(frame, r))
            video.write(img)
        else:
            video.write(img)
        frame_num += 1

    cv2.imshow('img', img)
    cv2.waitKey(0)

    cap.release()
    video.release()
    cv2.destroyAllWindows()
    print('-'*50+'finished generating video'+'-'*50)


# 边缘检测
t = 30  # threshold 阈值 (t越大，边缘差距就需要越大), default = 20
r = 3.5  # SUSAN 模板近似半径, default = 3.5
md = int(ceil(r * 2))  # SUSAN 模板近似尺寸
per_n_frame = 10  # 每n帧捕捉一次

dir = input('video directory:')
transform(dir)