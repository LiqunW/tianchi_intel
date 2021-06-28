# coding:utf-8
import os
import csv
import multiprocessing

out_dir = '../dataset/img_data/test/images'
os.makedirs(out_dir, exist_ok=True)
out_label_dir = '../dataset/img_data/test/labels'
os.makedirs(out_label_dir, exist_ok=True)

def download_img(img_url):
    os.system('wget -P {} {} '.format(out_dir, img_url))
    # wget.download(img_url, out=out_dir)
    
def parse_csv(csv_path):
    res = []
    label_res = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = csv.reader(f)
        for row in rows:
            try:
                tmp_dic = eval(row[1])
                img_url = tmp_dic['tfspath']
                img_name = img_url.split('/')[-1]
                label = eval(row[-1])
                res.append(img_url)
                label_res[img_name] = label
            except:
                continue
    return res, label_res


def parse_csv_test(csv_path):
    res = []
    label_res = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = csv.reader(f)
        for row in rows:
            try:
                tmp_dic = eval(row[1])
                img_url = tmp_dic['tfspath']
                # img_name = img_url.split('/')[-1]

                res.append(img_url)

            except:
                continue
    return res, label_res

def save_label(label_info):
    for k, v in label_info.items():
        idx = 0
        with open(os.path.join(out_label_dir, k+'.txt'), 'w', encoding='utf-8') as f:
            tmp_list = []
            for k1, v1 in v[1].items():
                tmp_list.append(k1)
                tmp_list.append(v1)
            tmp_str = '\t'.join(tmp_list)
            tmp_str = str(idx) + '\t' + tmp_str
            f.write(tmp_str+'\n')
            idx += 1
            # 标签和坐标
            for info in v[0]:
                txt_info = eval(info['text'])
                try:
                    tmp_line = str(idx) + '\t' + ','.join(info['coord']) + '\t' + \
                               txt_info['text'] + '\t' + txt_info['direction'] + '\n'
                except:
                    tmp_line = str(idx) + '\t' + ','.join(info['coord']) + '\t' + \
                               txt_info['text'] + '\t' + 'NoDirection' + '\n'
                f.write(tmp_line)
                idx += 1


if __name__ == "__main__":
    
    img_url, labels = parse_csv_test(r'../dataset/csv_data/test/Xeon1OCR_round1_test3_20210528.csv')
    
    # save_label(labels)
    
    with multiprocessing.Pool(processes=6) as p:
         p.map(download_img, img_url)