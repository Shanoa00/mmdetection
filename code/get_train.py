import pandas as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


train_path = '/home/mauricio/Documents/Pytorch/mmdetection/mmdetection_mau/data/HanDataset/' #S05_Detection&Recognition
folder = 'train_images' #S05_img
out = train_path+'test_images/'

# os.rename(train_path+'All_data_test.csv',train_path+'sample_submission.csv')
df = pd.read_csv(train_path+'sample_submission.csv') #sample_submission

os.makedirs(out,exist_ok=True)

print('Test ', len(df))
print('-----------')
# section, _ = train_test_split(df, test_size=0.653)
# print('New section ', len(section))

# section.to_csv(out+"sample_submission.csv", index=False)
img_ids = df['image_id']
# i=0
errors= []
for image in tqdm(img_ids):
    filename = image #+'.jpg'
    img = cv2.imread(os.path.join(train_path+folder,filename))
    try:
        cv2.imwrite(out+filename.split('.')[0]+'.jpg', img)
    except Exception as e:
        errors.append(filename)
        df.drop(df[df['image_id']==image].index, inplace=True)
    # i+=1
    # if i==10:
    #     break

print(errors)
df.to_csv(train_path+'new_sample_submission.csv', index=False)
# print(df.head(10))
# cv2.imwrite(out+filename, img)
    
# os.rename(train_path+folder,train_path+'train_images')

print('Done!')
