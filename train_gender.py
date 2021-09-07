# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:46:42 2021

@author: suman
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# initial parameters
epochs = 100 #number of cycles
lr = 1e-3 #learning rate
batch_size = 64 #number of images sent in a batch
img_dims = (96,96,3)#image dimension , 3 is channel rgb

data = []#append all images here
labels = []#man(0) or woman(1)

# load image files from the dataset
#glob module is used for easier fetching of all images from path. **->man/woman folder *->images.When recursive is set True “ ** ” followed by path separator ('./**/') will match any files or directories
image_files = [f for f in glob.glob(r'C:\Files\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files) #glob stores all man images first and then images . so if unshuffled model will learn all man images and then woman which spoils the accuracy

# converting images to arrays and labelling the categories
for img in image_files:

    image = cv2.imread(img)#read image
    #we need uniform size for all images
    image = cv2.resize(image, (img_dims[0],img_dims[1]))#resize as per dimensions set aboce that is 96x96
    image = img_to_array(image)#convert image to array
    data.append(image)#inside data list image array is appended
    
    #split breaks down path, and from back woman/man will be at -2 . this id done to know which image is man/woman
    label = img.split(os.path.sep)[-2] # C:\Files\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0
    #put the image category index in labels LIST 
    labels.append([label]) # [[1], [0], [0], ...]

# pre-processing
#for deep learning we need arrays, so convert image to array
data = np.array(data, dtype="float") / 255.0 #every image will have value from 0-255, so we divide by 255 to get smaller value(0-1) which is easier to compute 
#list to array
labels = np.array(labels)

# split dataset for training and validation

#20% to test, 80% for training
# random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls
#if no randaom state , different dataset each time
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

#this step gives two neuron in our model and easier to process
#left digit is man,right is woman . which ever is true that is 1
trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
#rotation_range: Int. Degree range for random rotations.
#width_shift_range: Float, 1-D array-like or int - float: fraction of total width, if < 1, or pixels if >= 1. - 1-D array-like: random elements from the array. - int: integer number of pixels from interval (-width_shift_range, +width_shift_range) - With width_shift_range=2 possible values are integers [-1, 0, +1], same as with width_shift_range=[-1, 0, +1], while with width_shift_range=1.0 possible values are floats in the interval [-1.0, +1.0).
#height_shift_range: Float, 1-D array-like or int - float: fraction of total height, if < 1, or pixels if >= 1. - 1-D array-like: random elements from the array. - int: integer number of pixels from interval (-height_shift_range, +height_shift_range) - With height_shift_range=2 possible values are integers [-1, 0, +1], same as with height_shift_range=[-1, 0, +1], while with height_shift_range=1.0 possible values are floats in the interval [-1.0, +1.0).
#shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
#zoom_range: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
#horizontal_flip: Boolean. Randomly flip inputs horizontally.
#fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode: - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k) - 'nearest': aaaaaaaa|abcd|dddddddd - 'reflect': abcddcba|abcd|dcbaabcd - 'wrap': abcdabcd|abcd|abcdabcd

# define model
def build(width, height, depth, classes):#(width of image,height of image,depth is rgb,number of categories->here 2 man and woman)
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1 #normalising, to keep value in datset to common scale . above depth seems to be first argument in some model . so to fix that
    #batch normalisation is done keep mean activation close to 0 and std deviation to 1
    
    #check channel
    if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first", 
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))#basically to reduce noise
    model.add(Dropout(0.25))#to avoid overfitting. 25% of neurons are deactivated during front and back propogation

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    #2d to 1d
    model.add(Flatten())

    model.add(Dense(1024))#1024->neurons, dense is fully connected layer. this is last layers
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))#final output are two classes , men or women
    model.add(Activation("sigmoid"))#either use softmax or sigmoid as they are probability based functions

    return model

# build model
#img_dims is above
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
#Lr->learning rate
#decay is an additional term in the weight update rule that causes the weights to exponentially decay to zero, if no other update is scheduled.
opt = Adam(lr=lr, decay=lr/epochs)

#specify the training configuration 
# Optimizer
# Loss function to minimize
# List of metrics to monitor
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])#optimiser decides how weight should be updated during back propogation

# train the model
#aug.flow replaces the original batch with the new, randomly transformed batch.
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)
# -> generator : a generator whose output must be a list of the form:
#                       - (inputs, targets)    
#                       - (input, targets, sample_weights)
# a single output of the generator makes a single batch and hence all arrays in the list 
# must be having the length equal to the size of the batch. The generator is expected 
# to loop over its data infinite no. of times, it should never return or exit.
# -> steps_per_epoch : it specifies the total number of steps taken from the generator
#  as soon as one epoch is finished and next epoch has started. We can calculate the value
# of steps_per_epoch as the total number of samples in your dataset divided by the batch size.
# -> Epochs : an integer and number of epochs we want to train our model for.
# -> Verbose : specifies verbosity mode(0 = silent, 1= progress bar, 2 = one line per epoch).
# -> callbacks : a list of callback functions applied during the training of our model.
# -> validation_data can be either:
#                       - an inputs and targets list
#                       - a generator
#                       - an inputs, targets, and sample_weights list which can be used to evaluate
#                         the loss and metrics for any model after any epoch has ended.
# -> validation_steps :only if the validation_data is a generator then only this argument
# can be used. It specifies the total number of steps taken from the generator before it is 
# stopped at every epoch and its value is calculated as the total number of validation data points
# in your dataset divided by the validation batch size.




# save the model to disk
model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')