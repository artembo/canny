import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve


class CannyEdgeDetector:
    def __init__(self, image, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, low_threshold=0.05, high_threshold=0.15):
        self.img = image
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def gaussian_kernel(self):
        size = int(self.kernel_size) // 2
        x, y = np.mgrid[-size: size + 1, -size: size + 1]
        normal = 1 / (2.0 * np.pi * self.sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * self.sigma ** 2))) * normal
        return g

    def sobel_filter(self, img):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        ix = ndimage.filters.convolve(img, kx)
        iy = ndimage.filters.convolve(img, ky)

        g = np.hypot(ix, iy)
        g = g / g.max() * 255
        theta = np.arctan2(iy, ix)
        return g, theta

    def non_max_suppression(self, img, d):
        m, n = img.shape
        z = np.zeros((m, n), dtype=np.int32)
        angle = d * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if img[i, j] >= q and img[i, j] >= r:
                        z[i, j] = img[i, j]
                    else:
                        z[i, j] = 0

                except IndexError:
                    pass

        return z

    def threshold(self, img):

        high_threshold = img.max() * self.high_threshold
        low_threshold = high_threshold * self.low_threshold

        m, n = img.shape
        res = np.zeros((m, n), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= high_threshold)

        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, image):
        img = image.copy()
        m, n = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if img[i, j] == weak:
                    try:
                        if (
                                img[i + 1, j - 1] == strong or
                                img[i + 1, j] == strong or
                                img[i + 1, j + 1] == strong or
                                img[i, j - 1] == strong or
                                img[i, j + 1] == strong or
                                img[i - 1, j - 1] == strong or
                                img[i - 1, j] == strong or
                                img[i - 1, j + 1] == strong
                        ):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError:
                        pass

        return img

    def detect(self):
        img_smoothed = convolve(self.img, self.gaussian_kernel())
        gradient_mat, theta_mat = self.sobel_filter(img_smoothed)
        non_max_img = self.non_max_suppression(gradient_mat, theta_mat)
        threshold_img = self.threshold(non_max_img)
        img_final = self.hysteresis(threshold_img)
        return img_final

    def get_all_stages(self):
        img_smoothed = convolve(self.img, self.gaussian_kernel())
        gradient_mat, theta_mat = self.sobel_filter(img_smoothed)
        non_max_img = self.non_max_suppression(gradient_mat, theta_mat)
        threshold_img = self.threshold(non_max_img)
        img_final = self.hysteresis(threshold_img)
        return [
            ('Сглаживание (фильтр Гаусса)', img_smoothed),
            ('Поиск градиентов', gradient_mat),
            ('Подавление немаксимумов', non_max_img),
            ('Двойная пороговая фильтрация', threshold_img),
            ('Гистерезис', img_final)
        ]
