import numpy as np
import cv2


def draw_points(img, points, r=3):
    '''
    img   : 用numpy数组表示的图片
    points: n*2的numpy数组
    '''

    # 在RGB空间网格状的生成颜色
    N = points.shape[0]
    I = int(N ** (1/3))
    J = I + 1 if I**3 < N else I
    K = I + 1 if I**2 * (I+1) < N else I
    I = I + 1 if I * (I+1)**2 < N else I
    color_i = []
    for i in range(I-1, -1, -1):
        color_i.append(int(255 * i / (I-1)))
    color_j = []
    for j in range(J-1, -1, -1):
        color_j.append(int(255 * j / (J-1)))
    color_k = []
    for k in range(K-1, -1, -1):
        color_k.append(int(255 * k / (K-1)))
    color = []
    for i in range(I):
        for j in range(J):
            for k in range(K):
                color.append((color_i[i], color_j[j], color_k[k]))

    for i in range(points.shape[0]):
        x, y = points[i]
        cv2.circle(img, (x,y), r, color[i], -1)
    return img
