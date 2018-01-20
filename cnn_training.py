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
                                          batch_size=32,
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
                                          batch_size=32,
                                          shuffle=True,
                                          seed=123)

test_image_generator = test_data_gen.flow_from_directory(directory="images/test",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=32,
                                          shuffle=True,
                                          seed=123)

# setup model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=train_input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
    train_image_generator,
    steps_per_epoch=746 // 32,
    epochs=50,
    validation_data=test_image_generator,
    validation_steps= 244 // 32)

model.save_weights('first_try.h5')

















