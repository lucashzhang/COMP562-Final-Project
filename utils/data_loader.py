import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data(batch_size):
    train_ds, val_ds = import_data(batch_size)
    return train_ds, val_ds

def import_data(batch_size):

    #dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/dfsdata2/test3_data/jackFan/datasets/resized_train/',
        color_mode="rgb",
        labels="inferred",
        label_mode="categorical",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size = (256, 256),
        batch_size = batch_size,
        shuffle=True,
        #smart_resize=True,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/dfsdata2/test3_data/jackFan/datasets/resized_val/',
        color_mode="rgb",
        labels="inferred",
        label_mode="categorical",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size = (256, 256),
        batch_size = batch_size,
        #smart_resize=True,
    )
    
    """
    # create a data generator
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=True,
        featurewise_std_normalization=False,
        samplewise_std_normalization=True,
        channel_shift_range=255,
        fill_mode='nearest',
        horizontal_flip=True,
    )

    # load and iterate training dataset
    train_ds = datagen.flow_from_directory(
        '/dfsdata2/test3_data/jackFan/datasets/vgg_yoga/train/',
        class_mode='categorical',
        target_size=(256,256),
        batch_size=batch_size,
        smart_resize=True
    )
    
    # load and iterate validation dataset
    val_ds = datagen.flow_from_directory(
        '/dfsdata2/test3_data/jackFan/datasets/vgg_yoga/val',
        class_mode='categorical',
        target_size=(256,256),
        batch_size=batch_size,
        smart_resize=True
    )
    """

    return train_ds, val_ds

"""
def train_test_split(X):
    
    def is_test(x, y):
        return x % 4 == 0

    def is_train(x, y):
        return not is_test(x, y)

    recover = lambda x,y: y

    test_dataset = X.enumerate() \
                        .filter(is_test) \
                        .map(recover)

    train_dataset = X.enumerate() \
                        .filter(is_train) \
                        .map(recover)
    
    return train_dataset, test_dataset

def test_val_split(X):
    
    def is_test(x, y):
        return x % 2 == 0

    def is_val(x, y):
        return not is_test(x, y)

    recover = lambda x,y: y

    test_dataset = X.enumerate() \
                        .filter(is_test) \
                        .map(recover)

    val_dataset = X.enumerate() \
                        .filter(is_val) \
                        .map(recover)
    
    return test_dataset, val_dataset

def features_labels_split(X,Y,Z):
    
    X_train = np.concatenate([x for x, y in X], axis=0)
    Y_train = np.concatenate([y for x, y in X], axis=0)

    X_test = np.concatenate([x for x, y in Y], axis=0)
    Y_test = np.concatenate([y for x, y in Y], axis=0)
    
    X_val = np.concatenate([x for x, y in Z], axis=0)
    Y_val = np.concatenate([y for x, y in Z], axis=0)
    
    return X_train, Y_train, X_test, Y_test, X_val, Y_val
"""