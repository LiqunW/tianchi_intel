import cv2
import random
import PIL
from PIL import Image, ImageEnhance
import numpy as np

# 默认输出图片排列和输入相同

def brightened(img):
    if isinstance(img, PIL.Image.Image):
        enh_bri = ImageEnhance.Brightness(img)
        brightness = random.uniform(0.2, 1.2)
        image_brightened = enh_bri.enhance(brightness)
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enh_bri = ImageEnhance.Brightness(img)
        brightness = random.uniform(0.2, 1.2)
        image_brightened = enh_bri.enhance(brightness)
        image_brightened = cv2.cvtColor(np.asarray(image_brightened), cv2.COLOR_RGB2BGR)
    return image_brightened


def colored(img):
    if isinstance(img, PIL.Image.Image):
        enh_col = ImageEnhance.Color(img)
        color = random.uniform(0.5, 1.5)
        image_colored = enh_col.enhance(color)
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enh_col = ImageEnhance.Color(img)
        color = random.uniform(0.5, 1.5)
        image_colored = enh_col.enhance(color)
        image_colored = cv2.cvtColor(np.asarray(image_colored), cv2.COLOR_RGB2BGR)
    return image_colored


def contrasted(img):
    if isinstance(img, PIL.Image.Image):
        enh_con = ImageEnhance.Contrast(img)
        contrast = random.uniform(0.2, 1.2)
        image_contrasted = enh_con.enhance(contrast)
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enh_con = ImageEnhance.Contrast(img)
        contrast = random.uniform(0.2, 1.2)
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)
    return image_contrasted


def sharped(img):
    if isinstance(img, PIL.Image.Image):
        enh_sha = ImageEnhance.Sharpness(img)
        sharpness = random.uniform(0.2, 1.2)
        image_sharped = enh_sha.enhance(sharpness)
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enh_sha = ImageEnhance.Sharpness(img)
        sharpness = random.uniform(0.2, 1.2)
        image_sharped = enh_sha.enhance(sharpness)
        image_sharped = cv2.cvtColor(np.asarray(image_sharped), cv2.COLOR_RGB2BGR)
    return image_sharped


def random_hsv_transform(img):
    hue_delta = 1 + np.random.uniform(-0.3, 0.3)
    sat_mult = 1 + np.random.uniform(-0.3, 0.3)
    val_mult = 1 + np.random.uniform(-0.3, 0.3)
    if isinstance(img, PIL.Image.Image):
        img_hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        img = Image.fromarray(cv2.cvtColor(np.round(img_hsv).astype(np.uint8),cv2.COLOR_HSV2RGB))
        return img
    else:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        return cv2.cvtColor(np.round(img_hsv).astype(np.uint8),cv2.COLOR_HSV2BGR)


def warp_perspective_image(img):
    if isinstance(img, PIL.Image.Image):
        src = np.asarray(img)
    else:
        src = img
    height, width = src.shape[:2]
    shift_range = int(height*0.1)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = src_points.copy()
    for i in range(4):
        dst_points[i][0] += np.random.randint(0, shift_range, 1)
        dst_points[i][1] += np.random.randint(0, shift_range, 1)
    perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
    dst = cv2.warpPerspective(src, perspective_mat, (width, height), borderValue=(255, 255, 255, np.random.randint(0,5)))
    if isinstance(img, PIL.Image.Image):
        dst = Image.fromarray(dst)
    return dst

def smooth(img, char_size=30):
    if isinstance(img, PIL.Image.Image):
        image = np.asarray(img)
    else:
        image = img
    if np.random.choice([True, False]):
        size = int(char_size / 24)
        image = cv2.blur(image, (size*2+1, size*2+1))
    h, w = image.shape[:2]
    scale = 1 - np.random.random() * 0.2
    image = cv2.resize(image, (int(w * scale), int(h*scale)))
    image = cv2.resize(image, (w, h))
    if isinstance(img, PIL.Image.Image):
        image = Image.fromarray(image)
    return image


# todo 高斯模糊，下采样
def gaussian_blur(img):
    if isinstance(img, PIL.Image.Image):
        image = np.asarray(img)
    else:
        image = img
    g_kernel = random.randint(1, 5) * 2 + 1
    image = cv2.GaussianBlur(image, ksize=(g_kernel, g_kernel), sigmaX=0, sigmaY=0)
    image = np.uint8(image)
    if isinstance(img, PIL.Image.Image):
        image = Image.fromarray(image)
    return image


def down_up_sample(img):
    if isinstance(img, PIL.Image.Image):
        w, h = img.size
        img = img.resize((int(w / (random.random() * 2 + 1)), int(h / (random.random() * 2 + 1))),
                     Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    else:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w / (random.random() * 2 + 1)), int(h / (random.random() * 2 + 1))),
                         interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img

if __name__ == "__main__":
    img_path = r''
    a = Image.open(img_path)
    print()
