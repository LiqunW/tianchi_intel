import os
import tqdm

# 生成指定格式的gfile.txt文件

# update_test
def load_img_label(**data_path):
    img_paths = data_path['image']
    label_paths = data_path['label']
    def data_path_process():
        img_names = []
        label_names = []
        assert len(img_names) == len(label_names)
        for path_i, path_l in zip(img_paths, label_paths):
            images = os.listdir(path_i)
            labels = os.listdir(path_l)
            img_names.extend(list(map(lambda x: os.path.join(path_i, x), images)))
            label_names.extend(list(map(lambda x: os.path.join(path_l, x), labels)))
        img_names.sort(); label_names.sort()
        return img_names, label_names
    imgs, labels = data_path_process()
    return imgs, labels


def process_data(imgs, labels, out_dir='./gt.txt'):
    with open(out_dir, 'w', encoding='utf-8') as f:
        for img, l in zip(imgs, labels):
            with open(l, 'r', encoding='utf-8') as fr:
                tmp_l = fr.read()
            new_line = img + '\t' + tmp_l + '\n'
            f.write(new_line)


if __name__ == "__main__":
    data_path = {'image': [r'/mnt/jyyu/data/ocr/data_gen_100w/img',
                           r'/mnt/jyyu/data/ocr/data_gen_200w/img',
                           r'/mnt/jyyu/data/ocr/data_gen_200w_2/img',
                           r'/mnt/jyyu/data/ocr/data_gen_100w_random/img'
    
    ],
                 'label': [r'/mnt/jyyu/data/ocr/data_gen_100w/txt',
                           r'/mnt/jyyu/data/ocr/data_gen_200w/txt',
                           r'/mnt/jyyu/data/ocr/data_gen_200w_2/txt',
                           r'/mnt/jyyu/data/ocr/data_gen_100w_random/txt'
                 ]
    }
    #data_path = {'image':[r'/mnt/jyyu/data/ocr/xg_ctc_rs/test/JPEGImages'],
    #             'label':[r'/mnt/jyyu/data/ocr/xg_ctc_rs/test/TXT']
    #}
    imgs, labels = load_img_label(**data_path)
    process_data(imgs, labels, out_dir=r'./train_gen_gt.txt')
