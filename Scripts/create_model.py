from time import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from collections import Counter

EPOCHS = 10
BATCH_SIZE = 128

mnist = tf.keras.datasets.mnist 
(x_train,y_train) , (x_test,y_test) = mnist.load_data() 

x_train = tf.keras.utils.normalize(x_train,axis=1) 
x_test = tf.keras.utils.normalize(x_test,axis=1) 

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant')

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_validation = x_train[:12000]
y_validation = y_train[:12000]
x_train = x_train[12000:]
y_train = y_train[12000:]

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32, 1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))

model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'] 
            ) 

train_generator = ImageDataGenerator().flow(x_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(x_validation, y_validation, batch_size=BATCH_SIZE)

steps_per_epoch = x_train.shape[0]//BATCH_SIZE
validation_steps = x_validation.shape[0]//BATCH_SIZE

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, 
                    validation_data=validation_generator, validation_steps=validation_steps, 
                    shuffle=True, callbacks=[tensorboard])

val_loss,val_acc = model.evaluate(x_test,y_test) 
print("loss-> ",val_loss,"\nacc-> ",val_acc) 

path = os.path.dirname(os.path.dirname(__file__))
model.save(os.path.join(path, 'LeNet5.model')) 