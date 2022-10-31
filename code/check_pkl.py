import pickle
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, set_random_seed, train_detector

data_folder= 'Nancho_dataset/' #  kuzushiji

#train
train = open('../data/'+data_folder+'dtrain.pkl', 'rb')
val = open('../data/'+data_folder+'dtest.pkl', 'rb')
#test
#file= open('mmdetection/work_dirs/hr32/test_result.pkl', 'rb')

# dump information to that file
data = pickle.load(train)
val = pickle.load(val)

print(len(data))
for i in data:
    print(i)

# print("-------------------------")
# print(len(val))
# for i in val:
#     print(i['filename'])
#'umgy004-007.jpg'