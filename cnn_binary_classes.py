#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:16:35 2018

@author: angus
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

os.chdir("/home/angus/projects/monet_or_manet")

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
train_image_generator_binary = train_data_gen.flow_from_directory(directory="images_binary/train",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          class_mode='binary')

test_image_generator_binary = test_data_gen.flow_from_directory(directory="images_binary/test",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          class_mode='binary')

# and some for testing individual classes
test_manet_image_generator_binary = test_data_gen.flow_from_directory(directory="images/test_manet_only",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          class_mode='binary')

test_monet_image_generator_binary = test_data_gen.flow_from_directory(directory="images/test_monet_only",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          class_mode='binary')

test_sadio_mane_image_generator_binary = test_data_gen.flow_from_directory(directory="images/test_sadio_mane_only",
                                          target_size=(256, 256),
                                          color_mode='rgb',
                                          batch_size=batchSize,
                                          shuffle=True,
                                          seed=123,
                                          class_mode='binary')


train_data_size_binary = len(train_image_generator_binary.filenames)
test_data_size_binary = len(test_image_generator_binary.filenames)

# setup model structure
model = Sequential()

model.add(Conv2D(filters=1, kernel_size=(3, 3), input_shape=train_input_shape))
model.add(Activation('relu')) # 'Rectified Linear activation function' = max(0,x) or max(ax, x) where a<1
model.add(MaxPooling2D(pool_size=(2, 2))) # take max or average etc of a 'pool' of nodes to reduce dimensions of the output of a layer

model.add(Flatten()) # flattens the Conv2D output into big long vector
model.add(Dense(1))
model.add(Activation('sigmoid'))

optim_rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=optim_rmsprop,
              metrics=['accuracy'])

model.fit_generator(
        train_image_generator_binary,
        steps_per_epoch=train_data_size_binary // batchSize,
        epochs=10,
        validation_data=test_image_generator_binary,
        validation_steps=test_data_size_binary // batchSize,
        verbose=1
        )


preds_monet = model.predict_generator(generator=test_monet_image_generator_binary, steps=test_data_size_binary // batchSize)
print('Monet correct ', float(sum(preds_monet<0.5)/len(preds_monet)), '% of the time', sep=''),
print('MSE:', float(sum((1-preds_monet)**2)/len(preds_monet)))

preds_mane = model.predict_generator(generator=test_sadio_mane_image_generator_binary, steps=test_data_size_binary // batchSize)
print('Sadio Mane correct ', float(sum(preds_mane>0.5)/len(preds_mane)), '% of the time', sep=''),
print('MSE:', float(sum((1-preds_mane)**2)/len(preds_mane)))


model.get_weights()









