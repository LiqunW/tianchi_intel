#!/usr/bin/env python
# encoding: utf-8
'''
@author: stone
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: AUC_and_worry.py
@time: 2020/12/16 21:50
@desc:
'''
import os
import numpy as np
import shutil

def str_replace(str_):
    """
    替换字符
    :param str_:
    :return:
    """
    str_ = str_.replace(" ", "")
    str_ = str_.replace("：", ":")
    str_ = str_.replace("_", "-")
    return str_



dashed_line = '-' * 80
head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
disconnect = f'{dashed_line}\n{head}\n{dashed_line}'
data_dir = "/mnt/jyyu/data/ocr_train/test/JPEGImages/"
worry_name = "Attn_"
label_file = "all_label.txt"
demoR_file = "log_demo_Attn_result.txt"
label_dict = {}
result_dict = {}
if not os.path.exists(worry_name): os.makedirs(worry_name)
# 载入标签
with open(label_file, "r", encoding="utf-8") as f:
    label_all = f.readlines()
    for label_ in label_all:
        labe, str_ = label_.split(":")
        label_dict[labe] = str_
    f.close()
# 载入结果
with open(demoR_file, "r", encoding="utf-8") as f:
    label_all = f.readlines()
    for label_ in label_all:
        if label_ in disconnect:
            continue
        if len(label_.split("\t")) == 3:
            labe, str_, _ = label_.split("\t")
        else:
            labe, str_ = label_.split("\t")[0], ""
        result_dict[labe] = str_
    f.close()

result_keys = list(result_dict.keys())
tp = np.zeros(37) + 0.001
gt = np.zeros(37) + 0.001
with open(worry_name+".txt" , "a" ,encoding="utf-8") as f :
    f.writelines("图片名称\t真确结果\t错误结果\n")
    for i_keys in result_keys:
        label_n = int((i_keys.split("_")[-2])) - 1
        result_i = result_dict[i_keys].strip()
        gt_i = label_dict[i_keys.split("/")[-1]].strip()
        result_i = str_replace(result_i)
        gt_i = str_replace(gt_i)
        # 替换
        gt[label_n] += 1
        if result_i == gt_i:
            tp[label_n] += 1
        else:
            f.writelines("{}\t{}\t{}\n".format(i_keys.split("/")[-1], gt_i, result_i))
            img_dir = os.path.join(data_dir, i_keys.split("/")[-1])
            # shutil.copy(img_dir, worry_name)
    print(worry_name + "ACC", "---",  )
    print(list(tp/gt))
    print(sum(tp)/ sum(gt))



