import os
import shutil


# todo 去掉  ###的图片
with open(r'/work/dataset/train_special_gt.txt','r',encoding='utf-8') as f:
    lines = f.read().splitlines()

with open(r'/work/dataset/train_special_gt2.txt','w',encoding='utf-8') as f:
    for line in lines:
        path, label = line.split('\t')
        if label == "###":
            continue
        if "#" in label:
            label = label.replace('#', '')
        f.write(path+'\t'+label+'\n')
