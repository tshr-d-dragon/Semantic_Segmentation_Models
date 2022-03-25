import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


def conv_block(inputs, num_filters):

    x = Conv2D(num_filters, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x


def decoder_block(inputs, skip, num_filters):

    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    
    return x

def Unet_MobileNetV2(input_shape, classes = 1):             ## (512, 512, 3)

    """ Input """
    inputs = Input(shape=input_shape)
    
    """ Pre-trained MobileNetV2 """
    encoder = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs, alpha=1.0)
    
    """ Encoder """
    s1 = encoder.get_layer("input_1").output                ## (512 x 512)
    s2 = encoder.get_layer("block_1_expand_relu").output    ## (256 x 256)
    s3 = encoder.get_layer("block_3_expand_relu").output    ## (128 x 128)
    s4 = encoder.get_layer("block_6_expand_relu").output    ## (64 x 64)
    
    """ Bridge """
    b1 = encoder.get_layer("block_13_expand_relu").output   ## (32 x 32)
    
    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                         ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                         ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                         ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                          ## (512 x 512)
    
    """ Output """
    outputs = Conv2D(classes, (1, 1), padding="same", name="output_layer")(d4)
    
    if classes == 1:
      outputs = Activation('sigmoid')(outputs)
    else:
      outputs = Activation('softmax')(outputs)

    model = Model(inputs, outputs)
    
    return model

  
def main():

    model = Unet_MobileNetV2(input_shape=(512,512,3), classes=4)
    model.summary()


if __name__== '__main__':

    main()
