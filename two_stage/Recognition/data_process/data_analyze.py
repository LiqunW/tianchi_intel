import os
import tqdm
import cv2
from collections import Counter

# todo 统计标签长度

def process(img_path, label_path):
    label_len = []
    for label in tqdm.tqdm(os.listdir(label_path)):
        with open(os.path.join(label_path, label), 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        #img = cv2.imread(os.path.join(img_path, label.replace('.txt','')))
        labels = [l.split('\t')[2] for l in lines]
        label_len.extend([len(i) for i in labels])
    return label_len


if __name__ == "__main__":
    # img_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/common/images'
    # label_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/common/labels'
    #
    # len1 = process(img_path, label_path)
    #
    # img_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/special/images'
    # label_path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/special/labels'
    #
    # len2 = process(img_path, label_path)
    #
    # len_all = len1 # + len2
    #
    # len_c = Counter(len_all)
    # res = []
    # for k,v in len_c.items():
    #     res.append([k,v])
    # res.sort(key=lambda x:x[0],reverse=True)
    # import pickle
    # with open('label_len_common.pkl','wb') as f:
    #     pickle.dump(res,f)

    # 根据 pkl 分析
    import pickle
    with open('label_len_common.pkl','rb') as f:
        res = pickle.load(f)
    print()