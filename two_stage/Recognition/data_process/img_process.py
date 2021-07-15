import os
import tqdm


def load_data(img_path, label_path):
    images = os.listdir(img_path)
    labels = os.listdir(label_path)
    images = [x.split('.')[0] for x in images]
    img_set = set(images)
    labels = [x.split('.')[0] for x in labels]
    label_set = set(labels)

    for img in tqdm.tqdm(images):
        if img not in label_set:
            print('remove', img)
            os.remove(os.path.join(img_path, img))
    for lab in tqdm.tqdm(labels):
        if lab not in img_set:
            print('label', lab)
            os.remove(os.path.join(label_path, lab))


if __name__ == '__main__':
    img_path = r'/home/jj/xuegu/data_gen_zhucedengji/data/gen_test/img50000'
    label_path = r'/home/jj/xuegu/data_gen_zhucedengji/data/gen_test/txt50000'
    load_data(img_path, label_path)