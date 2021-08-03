import os
import shutil

import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dataset(phase):
    # new dataset to make which all in dir total_text
    images_path = 'total_text/{}_images'.format(phase)
    gts_path = 'total_text/{}_gts'.format(phase)
    list_path = 'total_text/{}_list.txt'.format(phase)
    mkdir(images_path)
    mkdir(gts_path)
    # raw dataset
    common_path = r'C:\Work\1_work\tianchi_intel\dataset\huawei\2021_5_data\train_label_common.json'  # "../data/{}_label_common.json".format(phase)
    special_path = r'C:\Work\1_work\tianchi_intel\dataset\huawei\2021_5_data\train_label_special.json'  # "../data/{}_label_special.json".format(phase)
    # info dict from raw dataset json
    with open(common_path, 'r' ,encoding="utf-8") as f:
        line_common = f.readlines()[0].strip()
    with open(special_path, 'r',encoding="utf-8") as f:
        line_special = f.readlines()[0].strip()
    # mix together
    json_dict = eval(line_common)
    json_dict.update(eval(line_special))
    # make new dataset
    f = open(list_path, 'w', encoding="utf-8")
    for image_name in json_dict:
        # list
        g = open(os.path.join(gts_path, '{}.txt'.format(image_name)), 'w',encoding="utf-8")
        f.write('{}\n'.format(image_name))
        # image
        if os.path.exists(os.path.join(r'C:\Work\1_work\tianchi_intel\dataset\huawei\2021_5_data\train_image_common', image_name)):
            real_path = r'C:\Work\1_work\tianchi_intel\dataset\huawei\2021_5_data\train_image_common'
        else:
            real_path = r'C:\Work\1_work\tianchi_intel\dataset\huawei\2021_5_data\train_image_special'
        shutil.copy(os.path.join(real_path, image_name), images_path)
        # gt
        text_info_list = json_dict[image_name]
        for text_info in text_info_list:
            label = text_info['label'].replace(',', '逗')
            points = np.array(text_info['points']).astype(np.int32)
            coors = list(points.reshape(-1))
            for coor in coors:
                g.write('{},'.format(coor))
            g.write("{}\n".format(label))
        g.close()
    f.close()


def main():
    phase_list = ['train']
    for phase in phase_list:
        make_dataset(phase)


if __name__ == "__main__":
    main()