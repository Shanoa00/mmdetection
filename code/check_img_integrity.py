import pandas as np
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image


train_path = '/workspace/mmdetection_mau/data/R01-2_Detection&Recognition/' #HanDataset, S05_Detection&Recognition
folder = 'test_images' #S05_img
#out = train_path+'test_images/'

# os.rename(train_path+'All_data_test.csv',train_path+'sample_submission.csv')
file= 'sample_submission.csv' if folder== 'test_images' else 'train.csv'
df = pd.read_csv(train_path+file) #train, sample_submission

#os.makedirs(out,exist_ok=True)

print(folder, len(df))
print('-----------')
# section, _ = train_test_split(df, test_size=0.653)
# print('New section ', len(section))

# section.to_csv(out+"sample_submission.csv", index=False)
img_ids = df['image_id']
# i=0
errors= []
for image in tqdm(img_ids):
    filename = image +'.jpg'
    # if filename=='79.jpg':
    try:
        im = Image.open(os.path.join(train_path+folder,filename))
        im.load()
        im.size
        ima= im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) #.show
        im.close()
    except Exception as e:
        # print(e)
        errors.append(filename)
        #df.drop(df[df['image_id']==image].index, inplace=True)
    # i+=1
    # if i==10:
    #     break

errors.sort()
print(errors)
print('Errors:',len(errors) )
# df.to_csv(train_path+'new_sample_submission.csv', index=False)
# print(df.head(10))
# cv2.imwrite(out+filename, img)
    
# os.rename(train_path+folder,train_path+'train_images')

print('Done!')
