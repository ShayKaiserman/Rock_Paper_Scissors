import cv2
import numpy as np
import keras
# from keras_squeezenet import SqueezeNet
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import os
import shutil

# ----- functions ------

def mapper(val):
    return CLASS_MAP[val]


def get_model():
    # imports the mobilenet model and discards the last 1000 neuron layer.
    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(4, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)

    return model

# img_save_path = os.getcwd()+'\\existing data\\Rock Paper Scissors.v1-raw-300x300\\train'

img_save_path = os.getcwd()+'\\collected data\\train'
test_path = os.getcwd()+'\\collected data\\test'

def split_data(img_save_path, test_path, N):
    # take random 20% from each category images and move to the 'test' directory
    # N=0.2
    for directory in os.listdir(img_save_path):
        source_folder = os.path.join(img_save_path, directory)
        dest_folder = os.path.join(test_path, directory)
        for f in os.listdir(source_folder):
            if np.random.rand(1) < N:
                shutil.move(source_folder + '\\' + f,
                            dest_folder + '\\' + f)
    print('data has been split')

# ----- script ------

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "None": 3
}

NUM_CLASSES = len(CLASS_MAP)

# load images from the directory
# split_data(img_save_path, test_path, 0.2)
dataset = []
for directory in os.listdir(img_save_path):
    path = os.path.join(img_save_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        print(img.shape)
        # resize the image to the original training input of MobileNet
        img = cv2.resize(img, (227, 227))
        print(img.shape)
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper'],
    ...
]
'''
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
print (labels)

'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels)
print (labels)

# define the model
model = get_model()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

# start training
model.fit(np.array(data), np.array(labels), epochs=10)

# save the model for later use
print("saving the trained model")
model.save("rock-paper-scissors-model-2.h5")
