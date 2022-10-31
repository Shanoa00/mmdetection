import pickle
import matplotlib.pyplot as plt
import PIL
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
import numpy as np

root_dir = '/home/mauricio/Documents/Pytorch/mmdetection/data/Nancho_dataset/'
path = '/home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/convnext_nancho_pre_kuzh/'
gt = open(path+'image_p.pkl', 'rb')
pred = open(path+'predic.pkl', 'rb')
#test
#file= open('mmdetection/work_dirs/hr32/test_result.pkl', 'rb')

# dump information to that file
img_p = pickle.load(gt)
pred = pickle.load(pred)
img = mmcv.imread(root_dir + 'test_images/'+img_p['filename'])
# print(len(pred[1]))
labs= np.ones(len(pred[1]),dtype=str)
# for i in range(len(pred[1])):
#     labs.append('c')  

# print(labs)
# print(len(pred[1]))
# print(len(labs))
image = mmcv.visualization.imshow_bboxes(img,img_p['ann']['bboxes'],thickness=2, show=False)
mmcv.visualization.imshow_det_bboxes(image, pred[1], labels= labs, thickness=2, 
                                     bbox_color='red',text_color='red',font_scale=0.02, 
                                     out_file=path+'shows/'+img_p['filename'])
                                     
# print("-------------------------")
# print(len(val))
# for i in val:
#     print(i['filename'])
#'umgy004-007.jpg'