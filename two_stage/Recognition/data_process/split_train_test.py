import os
from collections import defaultdict
import numpy as np

# 读取txt，分训练85%测试15%


def load_txt(txt_path=r'/home/jj/xuegu/ocr_recog/label'):
    txts = list(map(lambda x: os.path.join(txt_path,x), os.listdir(txt_path)))
    data = {}
    for txt in txts:
        with open(txt, 'r', encoding='utf-8') as f:
            lines = f.read().split()
        for l in lines:
            img, label = l.split(':')
            data[img] = label
    return data


def find_cls_num(data):
    # 找出有多少个字段 _x_y.png x为不同的字段
    data_by_cls = defaultdict(dict)
    for key, val in data.items():
        _, x, y = key.split('_')
        data_by_cls[x].update({key: val})

    # for key in data_by_cls.keys():
    #     print(key, len(data_by_cls[key]))

    return data_by_cls

def split_train_val(data, train_txt=r'train_gt.txt',test_txt=r'test_gt.txt',img_base=r'/home/jj/xuegu/ocr_recog/dataset'):
    with open(train_txt, 'w', encoding='utf-8') as f1,\
         open(test_txt, 'w', encoding='utf-8') as f2:
        for key, val in data.items():
            train_num = int(len(val) * 0.85)
            total_data = np.array([(k,v) for k,v in val.items()])
            random_num = np.random.permutation(len(val))
            train = total_data[random_num[:train_num]]
            test = total_data[random_num[train_num:]]
            for d in train:
                f1.write(os.path.join(img_base, d[0])+'\t'+d[1]+'\n')
            for d in test:
                f2.write(os.path.join(img_base, d[0])+'\t'+d[1]+'\n')


if __name__ == '__main__':
    data = load_txt()
    data = find_cls_num(data)
    split_train_val(data)
