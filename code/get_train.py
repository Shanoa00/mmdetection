import pandas as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = '/home/mauricio/Documents/Pytorch/mmdetection/data/kuzushiji_morpho2/'
folder = 'test_images'
out = '/home/mauricio/Documents/Pytorch/mmdetection/data/kuzushiji_morpho2_section/'
df = pd.read_csv(train_path+'sample_submission.csv')

# img_ids = df['image_id']
# print(img_ids.head())
# print(len(img_ids))
print('All ', len(df))
print('-----------')
section, _ = train_test_split(df, test_size=0.653)
print('New section ', len(section))

section.to_csv(out+"sample_submission.csv", index=False)
img_ids = section['image_id']

for image in img_ids:
    filename = image+'.jpg'
    img = cv2.imread(os.path.join(train_path+folder,filename))
    cv2.imwrite(out+folder+'/'+filename, img)
print('Done!')
