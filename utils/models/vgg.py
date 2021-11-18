import numpy as np
import tensorflow as tf
import h5py
from tensorflow import keras

def conv3_2(X, num_layers, block):
    """
    2x stacked convolutional layers. 
    
    Arguments:
    X -- tensor
    num_layers -- int
    block -- string

    Returns:
    X -- modified tensor
    """
    
    # CONV -> BN -> RELU Block applied to X - 1
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 2
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # MAXPOOL
    X = tf.keras.layers.MaxPooling2D((2, 2), name= block+'pool')(X)
    
    return X

def conv3_2_wo_bn(X, num_layers, block):
    """
    2x stacked convolutional layers. 
    
    Arguments:
    X -- tensor
    num_layers -- int
    block -- string

    Returns:
    X -- modified tensor
    """
    
    # CONV -> BN -> RELU Block applied to X - 1
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv1')(X)
    
    # CONV -> BN -> RELU Block applied to X - 2
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv2')(X)

    # MAXPOOL
    X = tf.keras.layers.MaxPooling2D((2, 2), name= block+'pool')(X)
    
    return X

def conv3_3(X, num_layers, block):
    """
    3x stacked convolutional layers. 
    
    Arguments:
    X -- tensor
    num_layers -- int
    block -- string

    Returns:
    X -- modified tensor
    """
    
    # CONV -> BN -> RELU Block applied to X - 1
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis = 1)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 2
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis = 1)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 3
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv3')(X)
    X = tf.keras.layers.BatchNormalization(axis = 1)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # MAXPOOL
    X = tf.keras.layers.MaxPooling2D((2, 2), name= block+'pool')(X)
    
    return X

def conv3_4(X, num_layers, block):
    """
    4x stacked convolutional layers. 
    
    Arguments:
    X -- tensor
    num_layers -- int
    block -- string


    Returns:
    X -- modified tensor
    """
    
    # CONV -> BN -> RELU Block applied to X - 1
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 2
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 3
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv3')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X - 4
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', kernel_regularizer=tf.keras.regularizers.l2(0.01), padding = 'same', strides = (1, 1), name = block+'conv4')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # MAXPOOL
    X = tf.keras.layers.MaxPooling2D((2, 2), name= block+'pool')(X)
    
    return X

def conv3_4_wo_bn(X, num_layers, block):
    """
    4x stacked convolutional layers. 
    
    Arguments:
    X -- tensor
    num_layers -- int
    block -- string


    Returns:
    X -- modified tensor
    """
    
    # CONV -> BN -> RELU Block applied to X - 1
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv1')(X)
    
    # CONV -> BN -> RELU Block applied to X - 2
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv2')(X)
    
    # CONV -> BN -> RELU Block applied to X - 3
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv3')(X)
    
    # CONV -> BN -> RELU Block applied to X - 4
    X = tf.keras.layers.Conv2D(num_layers, (3, 3), data_format='channels_last', activation = 'relu', padding = 'same', strides = (1, 1), name = block+'conv4')(X)

    # MAXPOOL
    X = tf.keras.layers.MaxPooling2D((2, 2), name= block+'pool')(X)
    
    return X

def vgg19_model(input_shape):
    """
    Implementation of VGG19.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
 
    X_input = tf.keras.Input(shape= (input_shape,input_shape,3))
    #X = tf.keras.layers.Dropout(0.1)(X_input)
    X = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(224,224),
    ]
    )(X_input)
    
    X = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(X)
    
    # First Block
    X = conv3_2(X, 64, "block1_")
    
    # Second Block
    X = conv3_2(X, 128, "block2_")
    
    # Third Block
    X = conv3_4(X, 256, "block3_")
    
    # Fourth Block
    X = conv3_4(X, 512, "block4_")
    
    #Fifth Block
    X = conv3_4(X, 512, "block5_")
    
    #Classification Block
    
    X = tf.keras.layers.Flatten(name='flatten')(X)
    X = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc1')(X)
    X = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc2')(X)
    X = tf.keras.layers.Dense(107, activation='relu', name='fc3')(X)
    X = tf.keras.layers.Dense(107, activation=tf.keras.activations.softmax, name='predictions')(X)
    model = tf.keras.Model(inputs = X_input, outputs = X, name='VGG19')
    
    model.load_weights('/dfsdata2/test3_data/jackFan/utils/weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    
    model.summary()
    return model

def vgg19_wo_bn_model(input_shape):
    """
    Implementation of VGG19.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
 
    X_input = tf.keras.Input(shape= (input_shape,input_shape,3))
    
    X = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(224,224),
    ]
    )(X_input)
    
    X = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(X)
    
    # First Block
    X = conv3_2_wo_bn(X, 64, "block1_")
    
    # Second Block
    X = conv3_2_wo_bn(X, 128, "block2_")
    
    # Third Block
    X = conv3_4_wo_bn(X, 256, "block3_")
    
    # Fourth Block
    X = conv3_4_wo_bn(X, 512, "block4_")
    
    #Fifth Block
    X = conv3_4_wo_bn(X, 512, "block5_")
    
    #Classification Block
    
    X = tf.keras.layers.Flatten(name='flatten')(X)
    X = tf.keras.layers.Dense(4096, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc1')(X)
    #X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc2')(X)
    #X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(107, activation=tf.keras.activations.softmax, name='predictions')(X)
    model = tf.keras.Model(inputs = X_input, outputs = X, name='VGG19_wo_bn')
    
    model.load_weights('/dfsdata2/test3_data/jackFan/utils/weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    
    model.summary()
    return model

def vgg16_model(input_shape):
    """
    Implementation of VGG16.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
 
    X_input = tf.keras.Input(shape= (input_shape,input_shape,3))
    
    X = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(224,224),
    ]
    )(X_input)
    
    X = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(X)
    
    # First Block
    X = conv3_2(X, 64, "block1_")
    
    # Second Block
    X = conv3_2(X, 128, "block2_")
    
    # Third Block
    X = conv3_3(X, 256, "block3_")
    
    # Fourth Block
    X = conv3_3(X, 512, "block4_")
    
    #Fifth Block
    X = conv3_3(X, 512, "block5_")
    
    #Classification Block
    
    X = tf.keras.layers.Flatten(name='flatten')(X)
    X = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc1')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='fc2')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(107, activation='relu', name='fc3')(X)
    X = tf.keras.layers.Dense(107, activation=tf.keras.activations.softmax, name='predictions')(X)
    model = tf.keras.Model(inputs = X_input, outputs = X, name='VGG16')
    
    model.load_weights('/dfsdata2/test3_data/jackFan/utils/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    
    model.summary()
    return model