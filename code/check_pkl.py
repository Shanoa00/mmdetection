import pickle
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, set_random_seed, train_detector
import itertools

data_folder= 'R01-2_Detection&Recognition/' #'HanDataset/', 'Nancho_dataset/',  kuzushiji
# classes_file= True

def unique(list1):
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def get_unique(data):
    unique1=[]
    for i in data:
        labels= i['ann']['labels']
        unique1.append(unique(labels))
    uniques= unique(list(itertools.chain.from_iterable(unique1)))
    
    # print(uniques)
    return uniques

def just_labels(data):
    labs=[]
    for i in data:
        labels= i['ann']['labels']
        labs.append(labels)
    return labs
    
#train
train = open('../data/'+data_folder+'dtrain.pkl', 'rb')  #dtrain_crop2.pkl
val = open('../data/'+data_folder+'dval.pkl', 'rb')
#test
test = open('../data/'+data_folder+'dtest.pkl', 'rb')


# dump information to that file
dtr = pickle.load(train)
dv = pickle.load(val)
dt = pickle.load(test)

#training
print('Training data:')
print(len(dtr),',', len(dv))

uni_train=get_unique(dtr)
uni_val=get_unique(dv)
print('only on val: ',len(set(uni_val)-set(uni_train)))

uni_train+= uni_val
comb= unique(uni_train)
print('unique labs:', len(comb))

print("-------------------------")
print('Testing data:')
print(len(dt))

uni_tes=get_unique(dt)
print('unique labs:', len(uni_tes))

# x = [1,2,3,4]
# f = [1,11,22,33,44,3,4]
#missing classes
miss_lab= set(uni_tes)-set(comb)
print(miss_lab)
print('only on test', len(miss_lab))

print("-------------------------")
all_d= uni_tes+ comb
print('unique labls All:', len(unique(all_d)))


# #Create the classes txt file
# if classes_file:
#     labs_train= just_labels(dtr)
#     labs_val= just_labels(dv)
#     all_comb= list(itertools.chain.from_iterable(labs_train+labs_val))
#     print(len(all_comb))
    
#     f= open('../data/'+data_folder+"classes.txt","w+")
#     # for i in all_comb:
#     for i in range(10000):
#         f.write(str(i)+'a'+'\n')

#     f.close()
#     print('classes.txt generated!')
# #'umgy004-007.jpg'