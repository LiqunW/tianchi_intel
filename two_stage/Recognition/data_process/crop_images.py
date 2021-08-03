import os
import tqdm
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
        tmp_box = np.array(tmp_box).astype(np.float32)
        img_crop = get_rotate_crop_image(image, tmp_box)
        img_crop_list.append(img_crop)
    return img_crop_list


def reorder_points(point_list):
    ordered_point_list = sorted(point_list, key=lambda x: (x[0], x[1]))
    first_point = ordered_point_list[0]
    slope_list = [[cal_slope(first_point, p), p] for p in ordered_point_list[1:]]
    ordered_slope_point_list = sorted(slope_list, key=lambda x: x[0])
    first_third_slope, third_point = ordered_slope_point_list[1]
    
    if above_line(ordered_slope_point_list[0][1], third_point, first_third_slope):
        second_point = ordered_slope_point_list[0][1]
        fourth_point = ordered_slope_point_list[2][1]
        reverse_flag = False
    else:
        second_point = ordered_slope_point_list[2][1]
        fourth_point = ordered_slope_point_list[0][1]
        reverse_flag = True
        
    second_fourth_slope = cal_slope(second_point, fourth_point)
    if first_third_slope < second_fourth_slope:
        if reverse_flag:
            reorder_point_list = [fourth_point, first_point, second_point, third_point]
        else:
            reorder_point_list = [second_point, third_point, fourth_point, first_point]
    else:
        reorder_point_list = [first_point, second_point, third_point, fourth_point]
    
    return reorder_point_list


def cal_slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-5)


def above_line(p, start_point, slope):
    y = (p[0] - start_point[0]) * slope + start_point[1]
    return p[1] < y

def save_res(imgs, labels, img_name):
    idx = 0
    for img, l in zip(imgs, labels):
        name = img_name.replace('.jpg', '_' + str(idx) + '.jpg')
        cv2.imwrite(os.path.join(OUTDIR_img, name), img)
        with open(os.path.join(OUTDIR_label, name+'.txt'), 'w', encoding='utf-8') as f:
            f.write(l)
        idx+=1

def points_convert(points):
    if len(points) != 8:
        points = np.array(points,dtype=np.int).reshape([-1,2])
        x,y,w,h = cv2.boundingRect(points)
        points = [x,y,x+w,y,x+w,y+h,x,y+h]
    return points

def process(img_path, label_path):
    for label in tqdm.tqdm(os.listdir(label_path)):
        with open(os.path.join(label_path, label), 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        img = cv2.imread(os.path.join(img_path, label.replace('.txt','')))
        labels = [l.split('\t')[2] for l in lines]
        points = [l.split('\t')[1] for l in lines]
        points = [p.split(',') for p in points]
        points = [[float(i) for i in p] for p in points]
        # process curve points
        points = [points_convert(p) for p in points]
        r_points = [reorder_points(np.array(p).reshape([4,2])) for p in points]
        imgs_crop = crop_images(img, r_points)
        save_res(imgs_crop, labels, label.replace('.txt',''))


if __name__ == "__main__":
    img_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/special/images'
    label_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/special/labels'
    OUTDIR_img = r'/work/crop/special/images'
    OUTDIR_label = r'/work/crop/special/labels'
    os.makedirs(OUTDIR_img, exist_ok=True)
    os.makedirs(OUTDIR_label, exist_ok=True)

    process(img_path, label_path)

