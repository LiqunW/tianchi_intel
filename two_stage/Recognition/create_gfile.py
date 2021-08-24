import os
import tqdm

# 生成指定格式的gfile.txt文件

# 改为生成器形式


class CreateGfile():
    def __init__(self, out_dir='./gt.txt', **data_path):
        self.fw = open(out_dir, 'w', encoding='utf-8')
        self.img_paths = data_path['image']
        self.label_paths = data_path['label']


    def data_path_process(self):
        img_names = []
        label_names = []
        assert len(img_names) == len(label_names)
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            for root, dirs, files in os.walk(img_path):
                for img in files:
                    img_ = os.path.join(root, img)
                    label_ = os.path.join(label_path, self.find_label(img))
                    print(img_, label_)
                    yield img_, label_


    def find_label(self, img):
        if img.endswith('jpg') or img.endswith('png'):
            label = img+'.txt'
        elif img.endswith('jpeg'):
            label = img+'.txt'
        else:
            assert False
        return label

    def read_label(self, label):
        with open(label, 'r', encoding='utf-8') as fr:
            l = fr.read()
        return l

    def __call__(self, *args, **kwargs):
        for img, label in self.data_path_process():
            label_txt = self.read_label(label)
            if '###' not in label_txt:
                new_line = img + '\t' + label_txt + '\n'
                self.fw.write(new_line)
        self.fw.close()


if __name__ == "__main__":
    data_path = {'image': [r'/work/dataset/crop/common/images',

    ],
                 'label': [r'/work/dataset/crop/common/labels',

                 ]
    }

    cg = CreateGfile(out_dir=r'/work/dataset/train_common_gt.txt', **data_path)
    cg()