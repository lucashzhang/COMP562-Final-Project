import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess_valid(valid_ds):
    
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ]
    )
    
    augmented_valid_ds = valid_ds.map(
        lambda x, y: (data_augmentation(x),y))
    
    return augmented_valid_ds

