import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import PIL
from PIL import Image
import numpy as np

def aug_seq():
    sometimes_5 = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes_8 = lambda aug: iaa.Sometimes(0.8, aug)
    sometimes_2 = lambda aug: iaa.Sometimes(0.2, aug)

    augmentations = iaa.Sequential([
        sometimes_2(iaa.CropAndPad(
            percent=(-0.02, 0.02),
            pad_mode=["edge"],
            pad_cval=(0, 255)
        )),
        iaa.Sequential([iaa.size.Resize(0.6), iaa.size.Resize(1 / 0.6)]),
        # sometimes_8(iaa.OneOf([iaa.Sequential([iaa.size.Scale(0.6),iaa.size.Scale(1/0.6)]),
        #                      iaa.Sequential([iaa.size.Scale(0.8),iaa.size.Scale(1/0.8)]),
        #                      iaa.Sequential([iaa.size.Scale(0.9),iaa.size.Scale(1/0.9)]),])),

        # This inverts completely inverts the text ( make black white etc.)
        sometimes_1(iaa.Invert(1, per_channel=True)),

        # This does some affine transformations
        sometimes_2(iaa.Affine(
            scale={"x": (0.8, 1), "y": (0.8, 1)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0, 0), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
            rotate=(-2, 2),  # rotate by -45 to +45 degrees
            shear=(-2, 2),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=["edge"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        sometimes_2(iaa.OneOf([
            iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(2, 3)),  # blur image using local means with kernel sizes between 2 and 7
            # iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
        ])),
        sometimes_2(iaa.Add((-10, 10), per_channel=0.5)),
        # change brightness of images (by -10 to 10 of original value)
        sometimes_8(iaa.AddToHueAndSaturation((-200, 200))),  # change hue and saturation
        sometimes_5(iaa.contrast.LinearContrast((0.8, 5), per_channel=0.5)),  # improve or worsen the contrast
        sometimes_1(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.2)),
        # move pixels locally around (with random strengths)
        # sometimes_1(iaa.PiecewiseAffine(scale=(0.001, 0.005))),
        sometimes_1(iaa.PiecewiseAffine(scale=(0.004, 0.008))),
    ])
    return augmentations

augmentations = aug_seq()

def mix_aug(image):
    flag = True
    if isinstance(image, tuple):
        if isinstance(image[0],PIL.Image.Image):
            image = tuple(map(lambda x:np.array(x), image))
        else:
            image = tuple(map(lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2RGB), image))
            flag = False
        aug_img = augmentations(images=image)
        if flag:
            aug_img = tuple(map(lambda x: Image.fromarray(x), aug_img))
        else:
            aug_img = tuple(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR), aug_img))
    else:
        if isinstance(image,PIL.Image.Image):
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            flag = False
        aug_img = augmentations(image=image)
        if flag:
            aug_img = Image.fromarray(aug_img)
        else:
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    return aug_img

if __name__ == "__main__":
    img_path = r'/workspace/code/1_work/tianchi_intel/two_stage/Recognition/demo_image/demo_10.jpg'
    img = Image.open(img_path)
    images = (img, img)
    res = mix_aug(images)
    ia.imshow(np.array(res[0]))