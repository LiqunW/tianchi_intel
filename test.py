import os



with open(r'/work/dataset/test_data_gt.txt','r',encoding='utf-8') as f:
    lines = f.read().splitlines()

with open(r'/work/dataset/rec_gt_test.txt', 'w', encoding='utf-8') as f:
    for l in lines:
        path, label = l.split('\t')
        path = os.path.split(path)[-1]
        path = os.path.join('test_data/rec/test',path)
        f.write(path+'\t'+label+'\n')