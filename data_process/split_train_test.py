import os
import shutil

# with open(r'/workspace/code/1_work/tianchi_intel/two_stage/Detection/East/train_data/icdar2015/text_localization/train_icdar2015_label.txt','r',encoding='utf-8') as f:
#     train_data = f.read().splitlines()
#     train_data = [i.split('\t')[0] for i in train_data]
#     train_data = [os.path.split(i)[-1].split('.')[0] for i in train_data]
#     train_data = set(train_data)
#
# train_out_path = r'/work/dataset/crop/train_data'
# os.makedirs(train_out_path, exist_ok=True)
#
# with open(r'/workspace/code/1_work/tianchi_intel/two_stage/Detection/East/train_data/icdar2015/text_localization/test_icdar2015_label.txt','r',encoding='utf-8') as f:
#     test_data = f.read().splitlines()
#     test_data = [i.split('\t')[0] for i in test_data]
#     test_data = [os.path.split(i)[-1].split('.')[0] for i in test_data]
#     test_data = set(test_data)
#
# test_out_path = r'/work/dataset/crop/test_data'
# os.makedirs(test_out_path, exist_ok=True)
#
# with open(r'/work/dataset/train_common_gt2.txt', 'r', encoding='utf-8') as f:
#     common_label = f.read().splitlines()
#
# with open(r'/work/dataset/test_data_gt.txt', 'a', encoding='utf-8') as f:
#     for cl in common_label:
#         name = cl.split('\t')[0]
#         name = os.path.split(name)[-1]
#         name = name.rsplit('_', 1)[0]
#         if name in test_data:
#             f.write(cl+'\n')
with open(r'/work/dataset/test_data_gt.txt', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
    path = [i.split('\t')[0] for i in lines]

for f in path:
    shutil.copy(f, r'/work/dataset/crop/test_data')