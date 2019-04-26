"""
University of Ottawa, Author: M.Hamed Mozaffari

BowNet and wBowNet models are implemented similar to the original UNET network,
 and they can be substituted by other models in your code. 
In Tensorflow session, x, y, and keep_prob should be feeded and logit is the 
output of the model. 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and cite our works in their research. 

"""

import tensorflow as tf
import numpy as np 

img_height = 256
img_width = 256
msk_height = 256
msk_width = 256

#%%
with tf.name_scope("BowNet"):
    def crop_and_concat(x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        # offsets for the top left corner of the crop
        offsets = [(x2_shape[1] - x1_shape[1]) // 2, (x2_shape[2] - x1_shape[2]) // 2]
        off_1 = int(np.floor(offsets[0].value))
        off_2 = int(np.ceil(offsets[1].value))
        return tf.keras.layers.concatenate([x1, tf.keras.layers.Cropping2D(cropping=((off_1, off_1), (off_2, off_2)))(x2)], axis=-1)
    
    x = tf.placeholder(tf.float32, [None, img_height, img_width, 3], name='x')
    y = tf.placeholder(tf.float32, [None, msk_height, msk_width, 1], name='y')
    keep_prob = tf.placeholder(tf.float32, name='drop_out')

    kernel_size = (5, 5)
    activation = 'relu'
    pad = 'same'
    
    # encoding Layer #1 
    conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_1")(x)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Dropout(keep_prob)(conv_1)
    conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_2")(conv_1)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Dropout(keep_prob)(conv_1)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_1) 

    # encoding Layer #2
    conv_D2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2")(conv_1)
    conv_D2 = tf.keras.layers.BatchNormalization()(conv_D2)
    conv_D2 = tf.keras.layers.Dropout(keep_prob)(conv_D2)
    conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_3")(pool_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Dropout(keep_prob)(conv_2)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2) 

    # encoding Layer #3
    conv_D4 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4")(conv_D2)
    conv_D4 = tf.keras.layers.BatchNormalization()(conv_D4)
    conv_D4 = tf.keras.layers.Dropout(keep_prob)(conv_D4)
    conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_4")(pool_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Dropout(keep_prob)(conv_3)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_3) 
    
    # encoding Layer #4
    conv_D8 = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, dilation_rate = 8, activation='relu', padding=pad, name="conv_D8")(conv_D4)
    conv_D8 = tf.keras.layers.BatchNormalization()(conv_D8)
    conv_D8 = tf.keras.layers.Dropout(keep_prob)(conv_D8)
    conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_5")(pool_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Dropout(keep_prob)(conv_4)
    
    # decoding Layer #1
    up_conv_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_1")(conv_4)
    if pad == 'same':
        up_conv_1_concate = tf.keras.layers.concatenate([up_conv_1, conv_3], axis=-1, name="up_conv_1_concate")
    elif pad == 'valid':
        up_conv_1_concate = crop_and_concat(up_conv_1, conv_3)
    up_conv_1_concate = tf.keras.layers.BatchNormalization()(up_conv_1_concate)
    up_conv_1_concate = tf.keras.layers.Dropout(keep_prob)(up_conv_1_concate)
       
    conv_D4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4_2")(conv_D8)
    conv_D4_2 = tf.keras.layers.BatchNormalization()(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Dropout(keep_prob)(conv_D4_2)
    
    conv_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_6")(up_conv_1_concate)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Dropout(keep_prob)(conv_5)

    # decoding Layer #2
    up_conv_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_2")(conv_5)
    if pad == 'same':
        up_conv_2_concate = tf.keras.layers.concatenate([up_conv_2, conv_2], axis=-1, name="up_conv_2_concate")
    elif pad == 'valid':
        up_conv_2_concate = crop_and_concat(up_conv_2, conv_2)
    up_conv_2_concate = tf.keras.layers.BatchNormalization()(up_conv_2_concate)
    up_conv_2_concate = tf.keras.layers.Dropout(keep_prob)(up_conv_2_concate)

    conv_D2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2_2")(conv_D4_2)
    conv_D2_2 = tf.keras.layers.BatchNormalization()(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Dropout(keep_prob)(conv_D2_2)

    conv_6 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_7")(up_conv_2_concate)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Dropout(keep_prob)(conv_6)

    # decoding Layer #3
    up_conv_3 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_3")(conv_6)
    if pad == 'same':
        up_conv_3_concate = tf.keras.layers.concatenate([up_conv_3, conv_1], axis=-1, name="up_conv_3_concate")
    elif pad == 'valid':
        up_conv_3_concate = crop_and_concat(up_conv_3, conv_1)   
    up_conv_3_concate = tf.keras.layers.BatchNormalization()(up_conv_3_concate)
    up_conv_3_concate = tf.keras.layers.Dropout(keep_prob)(up_conv_3_concate)

    conv_D1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, dilation_rate = 1, activation='relu', padding=pad, name="conv_D1_2")(conv_D2_2)
    conv_D1_2 = tf.keras.layers.BatchNormalization()(conv_D1_2)
    conv_D1_2 = tf.keras.layers.Dropout(keep_prob)(conv_D1_2)
    
    conv_7 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_8")(up_conv_3_concate)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Dropout(keep_prob)(conv_7)

    logits_concate = tf.keras.layers.concatenate([conv_D1_2, conv_7], axis=-1, name="logits_concatenation")
    
    logits = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding=pad, name="logits")(logits_concate)

#%%
with tf.name_scope("wBowNet"):
    def crop_and_concat(x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        # offsets for the top left corner of the crop
        offsets = [(x2_shape[1] - x1_shape[1]) // 2, (x2_shape[2] - x1_shape[2]) // 2]
        off_1 = int(np.floor(offsets[0].value))
        off_2 = int(np.ceil(offsets[1].value))
        return tf.keras.layers.concatenate([x1, tf.keras.layers.Cropping2D(cropping=((off_1, off_1), (off_2, off_2)))(x2)], axis=-1)
    
    x = tf.placeholder(tf.float32, [None, img_height, img_width, 3], name='x')
    y = tf.placeholder(tf.float32, [None, msk_height, msk_width, 1], name='y')
    keep_prob = tf.placeholder(tf.float32, name='drop_out')
    
    kernel_size = (3, 3)
    activation = 'relu'
    pad = 'same'
    
    # encoding Layer #1 
    conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_1")(x)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Dropout(keep_prob)(conv_1)
    conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_2")(conv_1)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Dropout(keep_prob)(conv_1)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_1) 
    
    # encoding Layer #2
    conv_D2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2")(conv_1)
    conv_D2 = tf.keras.layers.BatchNormalization()(conv_D2)
    conv_D2 = tf.keras.layers.Dropout(keep_prob)(conv_D2)
    conv_D2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2")(conv_D2)
    conv_D2 = tf.keras.layers.BatchNormalization()(conv_D2)
    conv_D2 = tf.keras.layers.Dropout(keep_prob)(conv_D2)
    conv_D2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2")(conv_D2)
    conv_D2 = tf.keras.layers.BatchNormalization()(conv_D2)
    conv_D2 = tf.keras.layers.Dropout(keep_prob)(conv_D2)
    conv_D2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2")(conv_D2)
    conv_D2 = tf.keras.layers.BatchNormalization()(conv_D2)
    conv_D2 = tf.keras.layers.Dropout(keep_prob)(conv_D2)
    
    conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_3")(pool_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Dropout(keep_prob)(conv_2)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2) 

    # encoding Layer #3
    conv_D4 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4")(conv_D2)
    conv_D4 = tf.keras.layers.BatchNormalization()(conv_D4)
    conv_D4 = tf.keras.layers.Dropout(keep_prob)(conv_D4)
    conv_D4 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4")(conv_D4)
    conv_D4 = tf.keras.layers.BatchNormalization()(conv_D4)
    conv_D4 = tf.keras.layers.Dropout(keep_prob)(conv_D4)
    conv_D4 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4")(conv_D4)
    conv_D4 = tf.keras.layers.BatchNormalization()(conv_D4)
    conv_D4 = tf.keras.layers.Dropout(keep_prob)(conv_D4)
    conv_D4 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4")(conv_D4)
    conv_D4 = tf.keras.layers.BatchNormalization()(conv_D4)
    conv_D4 = tf.keras.layers.Dropout(keep_prob)(conv_D4)
    
    conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_4")(pool_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Dropout(keep_prob)(conv_3)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_3) 
    
    # encoding Layer #4
    conv_D8 = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, dilation_rate = 8, activation='relu', padding=pad, name="conv_D8")(conv_D4)
    conv_D8 = tf.keras.layers.BatchNormalization()(conv_D8)
    conv_D8 = tf.keras.layers.Dropout(keep_prob)(conv_D8)
    
    conv_D4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4_2")(conv_D8)
    conv_D4_2 = tf.keras.layers.BatchNormalization()(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Dropout(keep_prob)(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4_2")(conv_D4_2)
    conv_D4_2 = tf.keras.layers.BatchNormalization()(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Dropout(keep_prob)(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4_2")(conv_D4_2)
    conv_D4_2 = tf.keras.layers.BatchNormalization()(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Dropout(keep_prob)(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, dilation_rate = 4, activation='relu', padding=pad, name="conv_D4_2")(conv_D4_2)
    conv_D4_2 = tf.keras.layers.BatchNormalization()(conv_D4_2)
    conv_D4_2 = tf.keras.layers.Dropout(keep_prob)(conv_D4_2)
    
    conv_D2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2_2")(conv_D4_2)
    conv_D2_2 = tf.keras.layers.BatchNormalization()(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Dropout(keep_prob)(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2_2")(conv_D2_2)
    conv_D2_2 = tf.keras.layers.BatchNormalization()(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Dropout(keep_prob)(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2_2")(conv_D2_2)
    conv_D2_2 = tf.keras.layers.BatchNormalization()(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Dropout(keep_prob)(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, dilation_rate = 2, activation='relu', padding=pad, name="conv_D2_2")(conv_D2_2)
    conv_D2_2 = tf.keras.layers.BatchNormalization()(conv_D2_2)
    conv_D2_2 = tf.keras.layers.Dropout(keep_prob)(conv_D2_2)
    
    conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_5")(pool_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Dropout(keep_prob)(conv_4)
    
    conv_4_c = crop_and_concat(conv_4, conv_D2_2)
    conv_3_c = crop_and_concat(conv_3, conv_D4_2)
    conv_2_c = crop_and_concat(conv_2, conv_D8)
    conv_1_c = crop_and_concat(conv_D2, conv_1)
    
    # decoding Layer #1
    up_conv_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_1")(conv_4_c)
    if pad == 'same':
        up_conv_1_concate = tf.keras.layers.concatenate([up_conv_1, conv_3_c], axis=-1, name="up_conv_1_concate")
    elif pad == 'valid':
        up_conv_1_concate = crop_and_concat(up_conv_1, conv_3_c)
    
    conv_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_6")(up_conv_1_concate)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Dropout(keep_prob)(conv_5)

    # decoding Layer #2
    up_conv_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_2")(conv_5)
    if pad == 'same':
        up_conv_2_concate = tf.keras.layers.concatenate([up_conv_2, conv_2_c], axis=-1, name="up_conv_2_concate")
    elif pad == 'valid':
        up_conv_2_concate = crop_and_concat(up_conv_2, conv_2_c)

    conv_6 = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_7")(up_conv_2_concate)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Dropout(keep_prob)(conv_6)

    # decoding Layer #3
    up_conv_3 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 2), padding=pad, strides=(2, 2), activation='relu', name="up_conv_3")(conv_6)
    if pad == 'same':
        up_conv_3_concate = tf.keras.layers.concatenate([up_conv_3, conv_1_c], axis=-1, name="up_conv_3_concate")
    elif pad == 'valid':
        up_conv_3_concate = crop_and_concat(up_conv_3, conv_1_c)    

    conv_7 = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=pad, name="conv_8")(up_conv_3_concate)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Dropout(keep_prob)(conv_7)
       
    logits = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding=pad, name="logits")(conv_7)