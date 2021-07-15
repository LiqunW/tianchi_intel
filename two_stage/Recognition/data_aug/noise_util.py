import PIL
from PIL import Image
import numpy as np
from scipy import ndimage


def speckle(src, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float) / 255
    severity = np.random.uniform(0, 0.2)
    # severity = 0.5
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape)*severity,1)
    blur_avg = np.average(blur[:, :, :3], axis=2)
    blur[:, :, 0] = blur_avg
    blur[:, :, 1] = blur_avg
    blur[:, :, 2] = blur_avg
    img_spec = (img + blur)
    img_spec[img_spec > 1] = 1
    img_spec[img_spec <= 0] = 0
    img_spec = img_spec * 255
    img_spec = img_spec.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img_spec = Image.fromarray(img_spec)
    return img_spec


def gaussian_noise(src, mean=0, sigma=20, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    sigma = np.random.randint(sigma)
    if noise_type == 1:
        noise_mat = np.random.normal(loc=mean, scale=sigma, size=img.shape)
    else:
        noise_mat = np.random.normal(loc=mean, scale=sigma, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def multiple_gaussian_noise(src, mean=0, alpha=0.1, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    alpha = np.random.random() * alpha
    if noise_type == 1:
        noise_mat = np.random.normal(loc=mean, scale=img*alpha, size=img.shape)
    else:
        noise_mat = np.random.normal(loc=mean, scale=np.average(img, axis=-1)*alpha, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def pepperand_salt_noise(src, percetage=0.05, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    percetage = np.random.random() * percetage
    # 盐噪声 椒噪声
    if noise_type == 1:
        noise_mask = np.random.choice([0, 1, 2], size=img.shape, p=[1-percetage, percetage/2, percetage/2])
    else:
        noise_mask = np.random.choice([0, 1, 2], size=img.shape[:2], p=[1-percetage, percetage/2, percetage/2])
        if len(img.shape) == 3:
            noise_mask = np.expand_dims(noise_mask, axis=-1)

    img = np.where(noise_mask==1, 255, img)
    img = np.where(noise_mask==2, 0, img)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def poisson_noise(src, lam=250, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    lam = np.random.randint(lam)
    if noise_type == 1:
        noise_mat = np.random.poisson(lam=lam, size=img.shape)
    else:
        noise_mat = np.random.poisson(lam=lam, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def lognormal_noise(src, mean=0, sigma=2.4, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    sigma = np.random.random() * sigma
    if noise_type == 1:
        noise_mat = np.random.lognormal(mean=mean, sigma=sigma, size=img.shape)
    else:
        noise_mat = np.random.lognormal(mean=mean, sigma=sigma, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def gamma_noise(src, shape=200, scale=1.0, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    shape = np.random.randint(shape)
    if noise_type == 1:
        noise_mat = np.random.gamma(shape=shape, scale=scale, size=img.shape)
    else:
        noise_mat = np.random.gamma(shape=shape, scale=scale, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img


def multiple_gamma_noise(src, alpha=0.6, scale=1.0, noise_type=1):
    if isinstance(src, PIL.Image.Image):
        img = np.array(src)
    else:
        img = src.copy()
    img = img.astype(np.float)
    alpha = np.random.random() * alpha
    if noise_type == 1:
        noise_mat = np.random.gamma(shape=img*alpha, scale=scale, size=img.shape)
    else:
        noise_mat = np.random.gamma(shape=np.average(img, axis=-1)*alpha, scale=scale, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    if isinstance(src, PIL.Image.Image):
        img = Image.fromarray(img)
    return img

if __name__ == '__main__':
    img_path = r''
    a = Image.open(img_path)
    print()