import cv2
import numpy as np


def mean_filter(input_image, filter_size=3):
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    new_image = cv2.blur(image, (filter_size, filter_size))
    return cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)


def gaussian_blur(input_image, filter_size=3):
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    new_image = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    return cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)


def noisy(noise_type, image):
    image = image / 255.0
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.05
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy[noisy < 0.0] = 0.0
        noisy[noisy > 1.0] = 1.0
        noisy = noisy * 255
        return noisy.astype(dtype=np.uint8)
    elif noise_type == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        out = out * 255
        return out.astype(dtype=np.uint8)
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy = noisy * 255
        return noisy.astype(dtype=np.uint8)
    elif noise_type == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        noisy[noisy < 0.0] = 0.0
        noisy[noisy > 1.0] = 1.0
        noisy = noisy * 255
        print(noisy)
        return noisy.astype(dtype=np.uint8)
