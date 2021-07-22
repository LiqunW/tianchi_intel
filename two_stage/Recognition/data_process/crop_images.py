import copy
import cv2
import numpy as np


def get_rotate_crop_image(img, points):
    img_crop_width = int(
        max(np.linalg.norm(points[0]-points[1]),
            np.linalg.norm(points[2]-points[3])
            )
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0]-points[3]),
            np.linalg.norm(points[1]-points[2])
            )
    )
    pts_std = np.float32([[0, 0],
                          [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def crop_images(image, dt_boxes):
    img_crop_list = []
    for idx in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[idx])
        tmp_box = tmp_box.astype(np.float32)
        img_crop = get_rotate_crop_image(image, tmp_box)
        img_crop_list.append(img_crop)
    return img_crop_list

def reorder_points(dt_boxes):
    return dt_boxes

def parse_txt()


if __name__ == "__main__":
    img_path = r'/workspace/tianchi_intel/dataset/huawei/processed/common/images'
    label_path = r''
    