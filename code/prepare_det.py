import mmcv
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def iter_bboxes(labels):
    if not labels:
        return
    labels = labels.split()
    n = len(labels)
    assert n % 5 == 0
    for i in range(0, n, 5):
        ch, x, y, w, h = labels[i:i + 5]
        yield int(x), int(y), int(w), int(h), ch

data_folder= 'kuzushiji_morpho2_section/'  #'Nancho_dataset/', 'kuzushiji/'

def prepare_train():
    df = pd.read_csv('../data/'+data_folder+'train.csv', keep_default_na=False)
    img_dir = Path('../data/'+data_folder+'train_images')

    unicode_translation = pd.read_csv('../data/'+data_folder+'unicode_translation.csv')
    unicode2class = dict(
        zip(unicode_translation['Unicode'], unicode_translation.index.values))

    # Add images to COCO
    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        image = {
            'filename': filename,
            'width': img.width,
            'height': img.height,
        }
        bboxes = []
        labels = []
        for x, y, w, h, ch in iter_bboxes(row['labels']):
            bboxes.append([x, y, x + w, y + h])
            labels.append(1)
            # labels.append(unicode2class[ch] + 1)
        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
        }
        images.append(image)

    import random
    train, val = train_test_split(images, test_size=0.119)
    print('All ', len(images))
    print('train ', len(train))
    print('val ', len(val))

    #random.shuffle(images)
    #mmcv.dump([im for im in images if im['filename'].startswith('umgy')], '../data/dval.pkl')
    #mmcv.dump([im for im in images if not im['filename'].startswith('umgy')], '../data/dtrain.pkl')

    mmcv.dump(val, '../data/'+data_folder+'dval.pkl')
    mmcv.dump(train, '../data/'+data_folder+'dtrain.pkl')
    mmcv.dump(images, '../data/'+data_folder+'dtrainval.pkl')


def prepare_test():
    df = pd.read_csv('../data/'+data_folder+'sample_submission.csv', keep_default_na=False)
    img_dir = Path('../data/'+data_folder+'test_images')

    # images = [] #Original code that doesn't add bbox and label to test dataset
    # for img_id, row in tqdm(df.iterrows()):
    #     filename = row['image_id'] + '.jpg'
    #     img = Image.open(img_dir / filename)
    #     images.append({
    #         'filename': filename,
    #         'width': img.width,
    #         'height': img.height,
    #         'ann': {
    #             'bboxes': np.array([], dtype=np.float32).reshape(-1, 4),
    #             'labels': np.array([], dtype=np.int64).reshape(-1, ),
    #             'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
    #             'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
    #         }
    #     })

    # Add images to COCO
    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        image = {
            'filename': filename,
            'width': img.width,
            'height': img.height,
        }
        bboxes = []
        labels = []
        for x, y, w, h, ch in iter_bboxes(row['labels']):
            bboxes.append([x, y, x + w, y + h])
            labels.append(1)
            # labels.append(unicode2class[ch] + 1)
        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
        }
        images.append(image)
    print('test ', len(images))
    mmcv.dump(images, '../data/'+data_folder+'dtest.pkl')


if __name__ == "__main__":
    prepare_train()
    prepare_test()
