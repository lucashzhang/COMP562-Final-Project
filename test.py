import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from utils import train
from utils import data_loader
from utils.models import resnet
from utils import valid

from tensorflow import keras

train_ds, val_ds = data_loader.load_data(64)
train_ds = train.preprocess_train(train_ds)

resnet50 = resnet.resnet50_model(256)
resnet50.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

resnet50.fit(train_ds,
epochs=10,verbose=1, validation_data=val_ds,
callbacks=[tensorboard_callback])

resnet50.save("resnet50_iteration1")