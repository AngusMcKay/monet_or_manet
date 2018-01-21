#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:16:35 2018

@author: angus
"""

import numpy
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# read data
mane_files = glob.glob("images/train/sadio_mane/*.jpg")
mane = ndimage.imread(mane_files[0])
plt.imshow(mane)
mane.shape[0]
plt.imshow(mane[0:300,0:300])

# general parameters
batchSize = 32

# setup data generators which will makes lots of copies of images with little tweaks
train_data_gen = ImageDataGenerator(
        #rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

test_data_gen = ImageDataGenerator(
        #rescale=1./255
        )

# test out augmented images
i = 0
for batch in train_data_gen.flow_from_directory(directory="images/train",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          save_to_dir="images/train_augmented",
                                          save_prefix="aug",
                                          save_format="jpeg"):
    
    i += 1
    if i > 20: # save 20 images
        break # otherwise generator will loop indefinitely

# test image shape to put into model
augmented_image = ndimage.imread('images/train_augmented/aug_0_4344528.jpeg')
train_input_shape = augmented_image.shape # (256, 256, 3) as expected

# setup flow_from_directory method from the ImageDataGenerator classes
train_image_generator = train_data_gen.flow_from_directory(directory="images/train",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123)

test_image_generator = test_data_gen.flow_from_directory(directory="images/test",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123)

# setup model
model = Sequential()
# can scan multiple filters over image for one layer, with different forms of averaging etc for each, then combine each with a different activation map
# can also have a 'stride' so that the filter doesn't take every possible combination, but only every second or third one
# can add initialization and regularization etc to layer if wanted
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=train_input_shape))
model.add(Activation('relu')) # 'Rectified Linear activation function' = max(0,x) or max(ax, x) where a<1
model.add(MaxPooling2D(pool_size=(2, 2))) # take max or average etc of a 'pool' of nodes to reduce dimensions of the output of a layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # flattens the Conv2D output into big long vector (of length 64 * 256 * 256 )
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # randomly sets the fraction of inputs to 0 at each update during training (i.e. drops out some nodes) which helps prevent overfitting
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



train_data_size = len(train_image_generator.filenames)
test_data_size = len(test_image_generator.filenames)

model.fit_generator(
    train_image_generator,
    # batch_size=32, # number of samples per gradient update (this is for 'fit' not 'fit_generator')
    steps_per_epoch=train_data_size // batchSize, # number of steps before declaring an epoch as completed (defaults to number of data points / batch size)
    epochs=50, # epoch is an iteration over the entire dataset
    # validation_split=0.2, # amount of input data to be held back for testing
    validation_data=test_image_generator,
    validation_steps= test_data_size // batchSize, # only relevant if steps_per_epoch set
    # callbacks,
    # class_weight, # dictionary mapping class indices to weights
    # sample_weight, # weighting for each training sample
    verbose=1 # 0=silent, 1=progress bar, 2=one line per epoch
    )

model.save_weights('first_try.h5')

















