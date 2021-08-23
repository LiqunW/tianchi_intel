import os
import pickle
from collections import Counter
import json
import tqdm

words_list = []

def parse_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    for key, val in tqdm.tqdm(labels.items()):
        for idx, v in enumerate(val):
            tmp_label = v['label']
            try:
                float(tmp_label)
            except:
                if tmp_label != "###" and len(tmp_label) > 1:
                    words = tmp_label.split(' ')
                    for w in words:
                        try:
                            float(w)
                        except:
                            if len(w) > 1 and '#' not in w:
                                words_list.append(w)

if __name__ == "__main__":
    parse_json(r'/workspace/code/1_work/tianchi_intel/dataset/huawei/Raw_data/train_label_common.json')
    parse_json(r'/workspace/code/1_work/tianchi_intel/dataset/huawei/Raw_data/train_label_special.json')
    cnt = Counter(words_list)
    with open(r'words_dict.pkl','wb') as f:
        pickle.dump(cnt,f)
    print()


