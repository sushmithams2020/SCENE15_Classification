import numpy as np
import os
import shutil
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
# from keras.utils import preprocess_input
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input
import random

def PreprocessDataset(dataset_path):

    train_path = dataset_path

    train_class_dir = os.listdir(train_path)

    train_img_data_list=[]
    train_labels = []
    for traindata in train_class_dir:
        image_list=os.listdir(train_path+'\\'+ traindata)
        print ('Loaded the images of dataset-'+'{}\n'.format(traindata))
        for img in image_list:
        
            img_path = train_path+'\\'+ traindata + '\\'+ img 
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            print('Input image shape:', x.shape)
            train_img_data_list.append(x)
            train_labels.append(traindata)

    train_img_data = np.array(train_img_data_list)
    train_img_data=np.rollaxis(train_img_data,1,0)
    train_img_data=train_img_data[0]

    Y_train = np_utils.to_categorical(train_labels, len(train_class_dir))

    #Shuffle the dataset
    total_data = list(zip(train_img_data,Y_train))
    random.shuffle(total_data)
    X_train,Y_train = zip(*total_data)

    return np.array(X_train),np.array(Y_train)

def LoadDataset():
    absFilePath = os.path.abspath(__file__)
    fileDir = os.path.dirname(absFilePath)
    file_loc = fileDir+"\\"+'15-Scene'
    train_path = file_loc+"\\Train"
    test_path = file_loc+"\\Test"
    X_train,Y_train = PreprocessDataset(train_path)
    X_test,Y_test = PreprocessDataset(test_path)

    return X_train,Y_train,X_test,Y_test

# Method to split Train and test data randomly. Used only for 1st time.
# def SplitTrainTestDataset_Scene15():
#     absFilePath = os.path.abspath(__file__)
#     fileDir = os.path.dirname(absFilePath)
#     file_loc = fileDir+"\\"+'15-Scene'
#     list_of_dir = os.listdir(file_loc)
#     class_dir = list_of_dir
#     for cls in class_dir:
#         if os.path.isdir(file_loc+"\\Train\\"+cls) is False:
#             os.makedirs(file_loc+"\\Train\\"+cls)
#         if os.path.isdir(file_loc+"\\Test\\"+cls) is False:
#             os.makedirs(file_loc+"\\Test\\"+cls)

#         src = file_loc+"\\"+cls
#         list_of_files = os.listdir(src)
#         np.random.shuffle(list_of_files)

#         num =0
#         for img in list_of_files:
#             if num != 100:
#                 shutil.move(file_loc+"\\"+cls+"\\"+img , file_loc+"\\Train\\"+cls)
#             else:
#                 break
#             num+=1

#         print("class=",cls) 
#         No_of_files = len(os.listdir(file_loc+"\\Train\\"+cls))
#         print(No_of_files)  

#         for test_img in os.listdir(src):
#             shutil.move(file_loc+"\\"+cls+"\\"+test_img , file_loc+"\\Test\\"+cls)
#         print("class=",cls) 
#         No_of_files = len(os.listdir(file_loc+"\\Test\\"+cls))
#         print(No_of_files) 

#     train_path = file_loc+"\\Train"
#     test_path = file_loc+"\\Test"

    # return train_path,test_path


