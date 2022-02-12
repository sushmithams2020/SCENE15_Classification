from keras.layers import Input,Dense,Flatten,GlobalAveragePooling2D,Dropout
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from glob import glob
from Scene15LoadDataset_TrainingFromScratch import LoadDataset
import time

def TrainFromScratchFineTuning():

    X_train,Y_train,X_test,Y_test = LoadDataset()
    imageinput = Input(shape=(224,224,3))
    resnetmodel = ResNet50(weights=None,include_top=False)
    last_layer = resnetmodel.output                 #last layer of resnet after removing the top layer

    x = GlobalAveragePooling2D()(last_layer)

    # add fully-connected & dropout layers
    x = Dense(512, activation='relu')(x)                #Fc-1
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)                #Fc-2
    x = Dropout(0.5)(x)

    num_classes  =15

    # softmax layer for classes
    predictions = Dense(num_classes, activation='softmax')(x)           #output layer

    model = Model(inputs=resnetmodel.input,outputs = predictions)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    start_time = time.time()
    
    model_result = model.fit(X_train, Y_train, batch_size=10, epochs=200, verbose=1, validation_data=(X_test, Y_test))
    training_accuracy = model_result.history['accuracy'][-1]
    
    end_time = time.time()
    print("Training time=",(end_time-start_time))
    print("Training Accuracy=",training_accuracy*100)

    (loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=10, verbose=1)
    print("Testing Accuracy=",accuracy*100)

TrainFromScratchFineTuning()




