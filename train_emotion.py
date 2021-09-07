# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:47:25 2021

@author: suman
"""

import matplotlib.pyplot as plt
import os

# Importing Deep Learning Libraries

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#Displaying Images

#common size for all images
picture_size = 48
folder_path = "../input/face-expression-recognition-dataset/images/"

expression = 'disgust'

plt.figure(figsize= (12,12))
for i in range(1, 10, 1):
    plt.subplot(3,3,i) #3x3 matrix kind of display
    img = load_img(folder_path+"train/"+expression+"/"+
                  os.listdir(folder_path + "train/" + expression)[i], target_size=(picture_size, picture_size))
    plt.imshow(img)   
plt.show()


#Making Training and Validation Data

#how many training example in one iteration
batch_size  = 128
#having training set and validation set
datagen_train  = ImageDataGenerator()
datagen_val = ImageDataGenerator()

train_set = datagen_train.flow_from_directory(folder_path+"train",#go inside training folder
                                              target_size = (picture_size,picture_size),#standardise the size
                                              color_mode = "grayscale",#for better accuracy
                                              batch_size=batch_size,#128
                                              class_mode='categorical',#7 different category of emotions
                                              shuffle=True)


# The directory must be set to the path where your ‘n’ classes of folders are present.
# The target_size is the size of your input images, every image will be resized to this size.
# color_mode: if the image is either black and white or grayscale set “grayscale” or if the image has three color channels, set “rgb”.
# batch_size: No. of images to be yielded from the generator per batch.
# class_mode: Set “binary” if you have only two classes to predict, if not set to“categorical”, in case if you’re developing an Autoencoder system, both input and the output would probably be the same image, for this case set to “input”.
# shuffle: Set True if you want to shuffle the order of the image that is being yielded, else set False.                                              


test_set = datagen_val.flow_from_directory(folder_path+"validation",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)


#Model Building



no_of_classes = 7 #7 different possible emotions

model = Sequential()#using sequential model

#1st CNN layer
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))#filter(detects pattern),kernel size
model.add(BatchNormalization())#to bring all the data to a common reference also to improve training speed and to balance large and small weights
model.add(Activation('relu'))#Activation is based on relu. Decides which data should pass to next layer. value of 0 or 1 is given
model.add(MaxPooling2D(pool_size = (2,2)))#Reduce the pixels by taking only best predicting pixels from every 2x2 matrix in previous whole pixel matrix
model.add(Dropout(0.25))#prevent from overfitting by randomly dropping some node. ie. learns training well, test not so good

#2nd CNN layer
model.add(Conv2D(128,(5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#3rd CNN layer
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#4th CNN layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())#2d to 1d image data because that is needed by ANN which starts next

#Fully connected 1st layer
model.add(Dense(256))#
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


# Fully connected layer 2nd layer
model.add(Dense(512))#fully connected, each node accepts input from all previous nodes. It provides learning feature from all combinations of features from previous layer
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(no_of_classes, activation='softmax'))#softmax activation rescales the model output so that it has the right properties



opt = Adam(lr = 0.0001)#Adam Optimiser. It is used to update model based  on loss function. Used to minimize loss
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])#metric is used to judge the performance
model.summary()


#Fitting the Model with Training and Validation Data


checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')#Checks model and saves the model. monitors validation accuracy. verbose is how the progress is shown to user.

early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )#If there is no improvement in validation accuracy. then stop it so that overfitting is not done

# Arguments

# monitor: Quantity to be monitored.
# min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
# patience: Number of epochs with no improvement after which training will be stopped.
# verbose: verbosity mode.
# mode: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.
# baseline: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
# restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used. An epoch will be restored regardless of the performance relative to the baseline. If no epoch improves on baseline, training will run for patience epochs and restore weights from the best epoch in that set.



reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)
                              #if my model is not able to cope with previous defined learning rate, then it reduces it

# Arguments

# monitor: quantity to be monitored.
# factor: factor by which the learning rate will be reduced. new_lr = lr * factor.
# patience: number of epochs with no improvement after which learning rate will be reduced.
# verbose: int. 0: quiet, 1: update messages.
# mode: one of {'auto', 'min', 'max'}. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.
# min_delta: threshold for measuring the new optimum, to only focus on significant changes.
# cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
# min_lr: lower bound on the learning rate.


#containins above parameter
callbacks_list = [early_stopping,checkpoint,reduce_learningrate]

#number of cycles
epochs = 48

# model.compile(loss='categorical_crossentropy',
#               optimizer = Adam(lr=0.001),
#               metrics=['accuracy'])


#TIme to fit model with training set
history = model.fit_generator(generator=train_set,
                                steps_per_epoch=train_set.n//train_set.batch_size,
                                epochs=epochs,
                                validation_data = test_set,
                                validation_steps = test_set.n//test_set.batch_size,
                                callbacks=callbacks_list
                                )

# -> generator : a generator whose output must be a list of the form:
#                       - (inputs, targets)    
#                       - (input, targets, sample_weights)
# a single output of the generator makes a single batch and hence all arrays in the list 
# must be having the length equal to the size of the batch. The generator is expected 
# to loop over its data infinite no. of times, it should never return or exit.
# -> steps_per_epoch : it specifies the total number of steps(batches of samples) taken from the generator
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


#Plotting Accuracy & Loss

plt.style.use('dark_background')

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()