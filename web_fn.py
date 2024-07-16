import cv2 as cv
import numpy as np


def cv2_cvt_color(img, cvt_code):
    return cv.cvtColor(img, eval(cvt_code))


def cv2_split(img, split_code):
    return img[:, :, split_code]


def cv2_split(img, split_code):
    return img[:, :, split_code]


def cv2_resize(img, width, height, fx, fy, resize_interpolation):
    return cv.resize(img, (width, height), fx=fx, fy=fy, interpolation=eval(resize_interpolation))


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


def cv2_pyr_down(img, width, height):
    return cv.pyrDown(img, dstsize=(width, height))


def cv2_sobel(show_image, kernel_size, x_alpha, y_alpha):
    sobel1 = cv.convertScaleAbs(cv.Sobel(show_image, cv.CV_64F, 0, 1, ksize=kernel_size))
    sobel2 = cv.convertScaleAbs(cv.Sobel(show_image, cv.CV_64F, 1, 0, ksize=kernel_size))
    rsSobel = cv.addWeighted(sobel1, x_alpha, sobel2, y_alpha, 0)
    # return np.hstack((sobel1, sobel2, rsSobel))
    return rsSobel


def cv2_scharr(show_image, x_alpha, y_alpha):
    sobel1 = cv.convertScaleAbs(cv.Scharr(show_image, cv.CV_64F, 0, 1))
    sobel2 = cv.convertScaleAbs(cv.Scharr(show_image, cv.CV_64F, 1, 0))
    rsSobel = cv.addWeighted(sobel1, x_alpha, sobel2, y_alpha, 0)
    # return np.hstack((sobel1, sobel2, rsSobel))
    return rsSobel


def cv2_laplacian(show_image, kernel_size):
    return cv.convertScaleAbs(cv.Laplacian(show_image, cv.CV_64F, ksize=kernel_size))


def cv2_canny(show_image, threshold1, threshold2):
    return cv.Canny(show_image, threshold1, threshold2)


def cv2_contours(show_image, contours_mode, contours_method):
    # 在进行过如转灰度图一类的操作后, 实际会处理为单通道图像, 但是由于显示组件始终处理为3通图, 故而做轮廓检测一类需要单通道图像时, 直接取0通道就好了
    # 如果直接传入彩图也可以处理,就转成灰度图
    # 传入灰度图也行,灰度图转灰度图也不会出错
    show_image_copy = show_image.copy()
    s2 = cv.cvtColor(show_image_copy, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(s2, eval(contours_mode), eval(contours_method))
    tt = cv.drawContours(show_image_copy, contours, -1, (255, 0, 0), 5)
    cnts = []
    for index, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        cnts.append(f"{index}:area={area}")
    return tt, str(cnts)
