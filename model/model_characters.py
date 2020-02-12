#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:36:43 2020

@author: ashi
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflowjs as tfjs

#initialising CNN
""" Creating the model """
classifier = Sequential()
# convolution layer 1
classifier.add(Convolution2D(64, (3, 3), input_shape=(28, 28, 3), activation="relu"))
# max pooling layer 1
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# convolution layer 2
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
# max pooling layer 2
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# flattening
classifier.add(Flatten())
# add the ANN
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=62, activation="softmax"))
# compiling
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/ashi/Desktop/CNN/English/Data/Training',
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/home/ashi/Desktop/CNN/English/Data/Testing',
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=2728,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=682)

#scores = classifier.evaluate(x_test, y_test, verbose=0)
#print("Error: {:.2f}%".format((1-scores[1])*100))

# save the model
tfjs.converters.save_keras_model(classifier, "model_characters")
