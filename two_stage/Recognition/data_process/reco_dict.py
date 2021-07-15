import os
from split_train_test import load_txt
import string
import pickle

# todo 识别字符集，统计真实label里的和3755汉字，英文数字


def cal_char_from_label(txt_path=None):
    data = load_txt(txt_path)
    char_set = set()
    for val in data.values():
        for v in val:
            char_set.add(v)
    print()
    return char_set


def add_txt_pkl_char(path, char_set):
    files = list(filter(lambda x: x.endswith('txt'), os.listdir(path)))
    for txt in files:
        if txt != 'chnidcard.txt':
            with open(os.path.join(path, txt), 'r', encoding='utf-8') as f:
                lines = f.read().split()
            for l in lines:
                for c in l:
                    if len(c) == 1:
                        char_set.add(c)
    # pkl = list(filter(lambda x: x.endswith('pkl'), os.listdir(path)))
    # for p_f in pkl:
    #     with open(os.path.join(path,p_f),'rb') as f:
    #         _, lines = pickle.load(f)
    #         for l in lines.keys():
    #             for c in l:
    #                 char_set.add(c)
    return char_set


def save_chars(char_set):
    print(len(char_set))
    with open('char_list.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(char_set))


if __name__ == '__main__':
    char_set = cal_char_from_label(txt_path=r'/home/jj/xuegu/ocr_recog/label') # 真实数据集label
    char_set = add_txt_pkl_char(r'/home/jj/xuegu/data_gen_zhucedengji/ocr/data/char', char_set) # 语料
    save_chars(char_set)
