import os
import shutil
import urllib
import csv

OUTDIR = '../dataset/img_data/test/images_rename'
os.makedirs(OUTDIR, exist_ok=True)

img_path = '../dataset/img_data/test/images'

def parse_csv_test(csv_path):
    res = []
    label_res = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = csv.reader(f)
        for row in rows:
            try:
                tmp_dic = eval(row[1])
                img_url = tmp_dic['tfspath']
                img_name = img_url.split('/')[-1]

                res.append(img_name)

            except:
                continue
    return res


def load_labels(path):
    raw_label_name = []
    for csv_f in [os.path.join(path, i) for i in os.listdir(path)]:
        tmp_res = parse_csv_test(csv_f)
        raw_label_name.extend(tmp_res)

    labels = [urllib.parse.unquote(i) for i in raw_label_name]
    return raw_label_name, labels


def rename_images(raw_label_name, labels):
    for raw_name, name in zip(raw_label_name, labels):
        try:
            old_img_name = name
            new_img_name = raw_name
            shutil.copy(os.path.join(img_path, old_img_name),
                        os.path.join(OUTDIR, new_img_name))
        except Exception as e:
            old_img_name = urllib.parse.unquote(name)
            new_img_name = raw_name
            shutil.copy(os.path.join(img_path, old_img_name),
                        os.path.join(OUTDIR, new_img_name))


if __name__ == "__main__":
    raw_label_name, labels = load_labels(r'../dataset/csv_data/test')
    rename_images(raw_label_name, labels)