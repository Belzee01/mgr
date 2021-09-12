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


def noise(noise_type, image):
    image = image / 255.0
    if noise_type == "gauss":
        mean = 0
        sigma = 0.05 ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        noised_image = image + gauss
        noised_image[noised_image < 0.0] = 0.0
        noised_image[noised_image > 1.0] = 1.0
        noised_image = noised_image * 255
        return noised_image.astype(dtype=np.uint8)
    elif noise_type == "s&p":
        s_p_factor = 0.5
        salt_factor = 0.004
        noised_image = np.copy(image)
        num_salt = np.ceil(salt_factor * image.size * s_p_factor)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noised_image[coords] = 1
        num_pepper = np.ceil(salt_factor * image.size * (1. - s_p_factor))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noised_image[coords] = 0
        noised_image = noised_image * 255
        return noised_image.astype(dtype=np.uint8)
    elif noise_type == "poisson":
        factor = len(np.unique(image))
        factor = 2 ** np.ceil(np.log2(factor))
        noised_image = np.random.poisson(image * factor) / float(factor)
        noised_image = noised_image * 255
        return noised_image.astype(dtype=np.uint8)
    elif noise_type == "speckle":
        gauss = np.random.randn(image.shape)
        gauss = gauss.reshape(image.shape)
        noised_image = image + image * gauss
        noised_image[noised_image < 0.0] = 0.0
        noised_image[noised_image > 1.0] = 1.0
        noised_image = noised_image * 255
        return noised_image.astype(dtype=np.uint8)
