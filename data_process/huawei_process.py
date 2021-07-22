import os
import json
import cv2
import tqdm


def parse_json(json_path, imgdir, outdir):
    with open(json_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
        
    for key, val in tqdm.tqdm(labels.items()):
        image = cv2.imread(os.path.join(imgdir, key), 1)
        cv2.imwrite(os.path.join(outdir, 'images', key.split('.')[0]+'.jpg'), image)
        with open(os.path.join(outdir, 'labels', key.split('.')[0]+'.jpg'+'.txt'), 'w',
                  encoding='utf-8') as f:
            for idx, v in enumerate(val):
                tmp_coord = ','.join([str(i) for i in sum(v['points'], [])])
                tmp_label = v['label']
                f.write(str(idx+1)+'\t'+tmp_coord+'\t'+tmp_label+'\n')
        

if __name__ == "__main__":
    common_dir = r'/workspace/tianchi_intel/dataset/huawei/processed/special'
    os.makedirs(os.path.join(common_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(common_dir, 'labels'), exist_ok=True)
    imgdir = r'/workspace/tianchi_intel/dataset/huawei/train_image_special'
    parse_json(r'/workspace/tianchi_intel/dataset/huawei/train_label_special.json',
               imgdir, common_dir)