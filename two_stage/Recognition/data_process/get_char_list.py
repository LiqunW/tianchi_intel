import os
import pickle
import tqdm

char_dict = set()

def parse_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    for line in lines:
        _, _, label = line.split('\t')
        for l in label:
            char_dict.add(l)


if __name__ == "__main__":
    path = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/common/labels'
    for label in tqdm.tqdm(os.listdir(path)):
        parse_txt(os.path.join(path, label))

    path2 = r'/workspace/code/1_work/tianchi_intel/dataset/huawei/processed/special/labels'
    for label in tqdm.tqdm(os.listdir(path2)):
        parse_txt(os.path.join(path2, label))

    with open('chars.txt','w',encoding='utf-8') as f:
        for c in char_dict:
            f.write(c + '\n')