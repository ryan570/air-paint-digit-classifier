import os

import numpy as np
import tensorflow as tf

path = os.path.dirname(os.path.dirname(__file__))
model = tf.keras.models.load_model(os.path.join(path, 'LeNet5.model'))

def predict(img):
    return np.argmax(model.predict(np.array([img])))

