"""
Common routines for openCV
"""
import cv2 as cv
import numpy as np
from PyCmpltrtok.common import sep, divide_int
import os


def my_pad(img, top=0, bottom=0, left=0, right=0):
    xmin = min(top, bottom, left, right)
    if xmin < 0:
        raise ValueError('Padding amount should not be negative.')
    ori_ch_slice = img.shape[2:3]
    h, w = img.shape[0:2]
    if top + bottom + left + right == 0:
        return img
    elif top + bottom == 0:
        img = np.concatenate([
            np.zeros((h, left, *ori_ch_slice), dtype=img.dtype),
            img,
            np.zeros((h, right, *ori_ch_slice), dtype=img.dtype),
        ], axis=1)
        return img
    elif left + right == 0:
        img = np.concatenate([
            np.zeros((top, w, *ori_ch_slice), dtype=img.dtype),
            img,
            np.zeros((bottom, w, *ori_ch_slice), dtype=img.dtype),
        ], axis=0)
        return img
    else:
        img = np.concatenate([
            np.zeros((h, left, *ori_ch_slice), dtype=img.dtype),
            img,
            np.zeros((h, right, *ori_ch_slice), dtype=img.dtype),
        ], axis=1)
        h, w = img.shape[0:2]
        img = np.concatenate([
            np.zeros((top, w, *ori_ch_slice), dtype=img.dtype),
            img,
            np.zeros((bottom, w, *ori_ch_slice), dtype=img.dtype),
        ], axis=0)
        return img


def imzoom2fit_rect(img, tgt_rect, is_padding=False, inter=cv.INTER_CUBIC):
    tgt_w, tgt_h = tgt_rect
    tgt_h, tgt_w = int(tgt_h), int(tgt_w)
    ori_h, ori_w = img.shape[:2]
    ori_ch_slice = img.shape[2:3]
    tgt_rate = tgt_w / tgt_h
    ori_rate = ori_w / ori_h
    if ori_rate < tgt_rate:
        if not is_padding:
            adj_h = tgt_h
            adj_w = tgt_h * ori_rate
            adj_w = int(round(adj_w))
            img = cv.resize(img, (adj_w, adj_h), interpolation=inter)
        else:
            adj_h = ori_h
            adj_w = adj_h * tgt_rate
            adj_w = int(round(adj_w))
            padding = abs(adj_w - ori_w)
            padding01, padding02 = divide_int(padding)
            img = my_pad(img, left=padding01, right=padding02)
            img = cv.resize(img, (tgt_w, tgt_h), interpolation=inter)
    else:
        if not is_padding:
            adj_w = tgt_w
            adj_h = adj_w / ori_rate
            adj_h = int(round(adj_h))
            img = cv.resize(img, (adj_w, adj_h), interpolation=inter)
        else:
            adj_w = ori_w
            adj_h = adj_w / tgt_rate
            adj_h = int(round(adj_h))
            padding = abs(adj_h - ori_h)
            padding01, padding02 = divide_int(padding)
            img = my_pad(img, top=padding01, bottom=padding02)
            img = cv.resize(img, (tgt_w, tgt_h), interpolation=inter)
    return img


if '__main__' == __name__:
    BASE_DIR, FILE_NAME = os.path.split(__file__)
    path = '/var/asuspei/large_data/pic/mat4zoom/burger_king.jpg'
    IMG_PATH = os.path.join(BASE_DIR, path)
    landscape = cv.imread(IMG_PATH, cv.IMREAD_COLOR)
    landscape_ = landscape.copy()
    print('landspace', landscape.shape)
    path = '/var/asuspei/large_data/pic/portrait/baby_in_car.jpg'
    IMG_PATH = os.path.join(BASE_DIR, path)
    portrait = cv.imread(IMG_PATH, cv.IMREAD_COLOR)
    portrait_ = portrait.copy()
    print('portrait', portrait.shape)

    tgt_rect = (400, 300)
    tgt_w, tgt_h = tgt_rect

    sep('zoom with paddings')
    landscape = imzoom2fit_rect(landscape_, tgt_rect, is_padding=True)
    print('landspace', landscape.shape)
    portrait = imzoom2fit_rect(portrait_, tgt_rect, is_padding=True)
    print('portrait', portrait.shape)
    row01 = np.concatenate((landscape, portrait), axis=1)

    sep('zoom without paddings')
    landscape = imzoom2fit_rect(landscape_, tgt_rect, is_padding=False)
    print('landspace', landscape.shape)
    landscape = my_pad(landscape, bottom=tgt_h - landscape.shape[0])
    print('landspace', landscape.shape)
    portrait = imzoom2fit_rect(portrait_, tgt_rect, is_padding=False)
    print('portrait', portrait.shape)
    portrait = my_pad(portrait, right=tgt_w - portrait.shape[1])
    print('portrait', portrait.shape)
    row02 = np.concatenate((landscape, portrait), axis=1)

    img = np.concatenate((row01, row02), axis=0)
    cv.line(img, (tgt_w, 0), (tgt_w, tgt_h * 2 - 1), (0, 255, 0), 1)
    cv.line(img, (0, tgt_h), (tgt_w * 2 - 1, tgt_h), (0, 255, 0), 1)
    cv.imshow('result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
