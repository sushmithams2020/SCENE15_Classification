from keras.layers import Input,Dense,Flatten,GlobalAveragePooling2D,Dropout
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from glob import glob
from LoadDataset import LoadDataset
import time

def TransferLearning():

    X_train,Y_train,X_test,Y_test = LoadDataset()
    num_classes  =15

    imageinput = Input(shape=(224,224,3))
    resnetmodel = ResNet50(input_tensor=imageinput,weights='imagenet',include_top=True)
    average_pool_layer = resnetmodel.get_layer('avg_pool').output

    x = Flatten()(average_pool_layer)
    predictions = Dense(num_classes, activation='softmax')(x)       #last layer

    model = Model(inputs=resnetmodel.input,outputs = predictions)

    for layer in resnetmodel.layers[:-1]:
        layer.trainable = False

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    start_time = time.time()
    model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_test, Y_test))
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time=",training_time)

    (loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
    print("Testing Accuracy=",accuracy*100)

TransferLearning()

