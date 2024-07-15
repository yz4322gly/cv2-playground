import cv2 as cv
import numpy as np


def cv2_cvt_color(img, cvt_code):
    return cv.cvtColor(img, eval(cvt_code))


def cv2_split(img, split_code):
    return img[:, :, split_code]


def cv2_split(img, split_code):
    return img[:, :, split_code]


def cv2_resize(img, width, height, fx, fy):
    return cv.resize(img, (width, height), fx=fx, fy=fy)


def cv2_roi(img, x1, y1, x2, y2):
    return img[x1:x2, y1:y2]


def cv2_erode(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.erode(img, kernel, iterations=iterations)


def cv2_dilate(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.dilate(img, kernel, iterations=iterations)


def cv2_threshold(img, thresh, maxval, threshold_type):
    _, t = cv.threshold(img, thresh, maxval, eval(threshold_type))
    return t


def cv2_open(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel=kernel, iterations=iterations)


def cv2_close(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=kernel, iterations=iterations)


def cv2_gradient(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel=kernel, iterations=iterations)


def cv2_tophat(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel=kernel, iterations=iterations)


def cv2_blackhat(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel=kernel, iterations=iterations)


def cv2_blur(img, kernel_size):
    return cv.blur(img, (kernel_size, kernel_size))


def cv2_box_filter(img, kernel_size, ddepth, normalize):
    return cv.boxFilter(img, ddepth, (kernel_size, kernel_size), normalize=normalize)


def cv2_gaussian_blur(img, kernel_size, sigmaX, dst, sigmaY):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), sigmaX, dst, sigmaY)


def cv2_median_blur(img, kernel_size):
    return cv.medianBlur(img, kernel_size)
