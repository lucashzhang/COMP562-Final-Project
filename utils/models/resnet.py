import numpy as np
import tensorflow as tf
import h5py
from tensorflow import keras

def identity_block50(X, c_channels, t_channels, stage, block):
    """
    Identity block of resnet50. 
    
    #Arguments:
        X: input tensor
        c_channels: size of continuation filters in main path
        t_channels: size of transition filters in main path
        stage: current stage
        block: current block
        
    #Returns:
        X - Identity block tensor
    """
    
    #naming conventions for imagenet weight compatibility
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'
    
    #shortcut save
    X_shortcut = X
    
    #main - first
    X = tf.keras.layers.Conv2D(c_channels, (1,1), kernel_initializer='glorot_normal', name=conv_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #main - second
    X = tf.keras.layers.Conv2D(c_channels, (3,3), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #main - third
    X = tf.keras.layers.Conv2D(t_channels, (1,1), kernel_initializer='glorot_normal', name=conv_base + '2c')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2c')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.add([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def conv_block50(X, c_channels, t_channels, stage, block):
    """
    Convolution block of resnet50. 
    
    #Arguments:
        X: input tensor
        size: CNN kernel size
        c_channels: size of continuation filters in main path
        t_channels: size of transition filters in main path
        stage: current stage
        block: current block
        
    #Returns:
        X - Convolution block tensor
    """
    
    #naming conventions for imagenet weight compatibility
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'

    #shortcut save
    X_shortcut = X
    
    #main - first
    X = tf.keras.layers.Conv2D(c_channels, (1, 1), strides=(2,2), kernel_initializer='glorot_normal', name=conv_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    #main - second
    X = tf.keras.layers.Conv2D(c_channels, (3,3), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    #main - third
    X = tf.keras.layers.Conv2D(t_channels, (1, 1), kernel_initializer='glorot_normal', name=conv_base + '2c')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2c')(X)

    #shortcut with conv
    shortcut = tf.keras.layers.Conv2D(t_channels, (1, 1), strides=(2,2), kernel_initializer='glorot_normal', name=conv_base + '1')(X_shortcut)
    shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '1')(shortcut)

    X = tf.keras.layers.add([X, shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def identity_block18(X, channels, stage, block):
    """
    Identity block of resnet18. 
    
    #Arguments:
        X: input tensor
        channels: size of filters in main path
        stage: current stage
        block: current block
        
    #Returns:
        X - Identity block tensor
    """
    
    #naming conventions for imagenet weight compatibility
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'
    
    #shortcut save
    X_shortcut = X
    
    #main - second
    X = tf.keras.layers.Conv2D(channels, (3,3), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #main - third
    X = tf.keras.layers.Conv2D(channels, (3,3), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.add([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def conv_block18(X, channels, stage, block, stride):
    """
    Convolution block of resnet18. 
    
    #Arguments:
        X: input tensor
        channels: size of filters in main path
        stage: current stage
        block: current block
        
    #Returns:
        X - Convolution block tensor
    """
    
    #naming conventions for imagenet weight compatibility
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'

    #shortcut save
    X_shortcut = X

    #main - second
    X = tf.keras.layers.Conv2D(channels, (3,3), strides=(stride,stride), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    #main - third
    X = tf.keras.layers.Conv2D(channels, (3,3), padding='same', kernel_initializer='glorot_normal', name=conv_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #shortcut with conv
    shortcut = tf.keras.layers.Conv2D(channels, (1, 1), strides=(stride,stride), kernel_initializer='glorot_normal', name=conv_base + '1')(X_shortcut)
    shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_base + '1')(shortcut)

    X = tf.keras.layers.add([X, shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def resnet18_model(input_shape):
    """
    Implementation of ResNet18
    
    #Arguments
        input_shape: size of input
    
    #Returns
        Instance of ResNet18
    """
    print('hhhhhhhhh')
    X_input = tf.keras.Input(shape= (input_shape,input_shape,3))
    
    #preprocessing
    X = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(224,224),
    ]
    )(X_input)
    
    X = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(X)
    
    
    #First Stage
    X = tf.keras.layers.Conv2D(64, (7,7), strides=(2,2), padding='same', kernel_initializer='glorot_normal',name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(X)
    
    #Second Stage
    X = conv_block18(X, 64, 2, 'a', 1)
    X = identity_block18(X, 64, 2, 'b')
    
    #Third Stage
    X = conv_block18(X, 128, 3, 'a', 2)
    X = identity_block18(X, 128, 3, 'b')
    
    #Fourth Stage
    X = conv_block18(X, 256, 4, 'a', 2)
    X = identity_block18(X, 256, 4, 'b')
    
    #Fifth Stage
    X = conv_block18(X, 512, 5, 'a', 2)
    X = identity_block18(X, 512, 5, 'b')
    
    #Classification Block
    X = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(X)
    X = tf.keras.layers.Dense(107, activation='softmax', name='fc1000')(X)
    
    model = tf.keras.Model(inputs = X_input, outputs = X, name='resnet18')
    
    model.load_weights('/dfsdata2/test3_data/jackFan/utils/weights/resnet18_imagenet_1000_no_top.h5', by_name=True)
    
    model.summary()
    
    return model

def resnet50_model(input_shape):
    """
    Implementation of ResNet50
    
    #Arguments
        input_shape: size of input
    
    #Returns
        Instance of ResNet50
    """
    
    X_input = tf.keras.Input(shape= (input_shape,input_shape,3))
    
    #preprocessing
    X = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(224,224),
    ]
    )(X_input)
    
    X = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(X)
    
    
    #First Stage
    X = tf.keras.layers.Conv2D(64, (7,7), padding='same', strides=(2,2),  kernel_initializer='glorot_normal',name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3,3), padding='same', strides=(2,2))(X)
    
    #Second Stage
    X = conv_block50(X, 64, 256, 2, 'a')
    X = identity_block50(X, 64, 256, 2, 'b')
    X = identity_block50(X, 64, 256, 2, 'c')
    
    #Third Stage
    X = conv_block50(X, 128, 512, 3, 'a')
    X = identity_block50(X, 128, 512, 3, 'b')
    X = identity_block50(X, 128, 512, 3, 'c')
    X = identity_block50(X, 128, 512, 3, 'd')
    
    #Fourth Stage
    X = conv_block50(X, 256, 1024, 4, 'a')
    X = identity_block50(X, 256, 1024, 4, 'b')
    X = identity_block50(X, 256, 1024, 4, 'c')
    X = identity_block50(X, 256, 1024, 4, 'd')
    X = identity_block50(X, 256, 1024, 4, 'e')
    X = identity_block50(X, 256, 1024, 4, 'f')
    
    #Fifth Stage
    X = conv_block50(X, 512, 2048, 5, 'a')
    X = identity_block50(X, 512, 2048, 5, 'b')
    X = identity_block50(X, 512, 2048, 5, 'c')
    
    #Classification Block
    X = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(X)
    X = tf.keras.layers.Dense(107, activation='softmax', name='fc1000')(X)
    
    model = tf.keras.Model(inputs = X_input, outputs = X, name='resnet50')
    
    model.load_weights('/dfsdata2/test3_data/jackFan/utils/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    
    model.summary()
    
    return model