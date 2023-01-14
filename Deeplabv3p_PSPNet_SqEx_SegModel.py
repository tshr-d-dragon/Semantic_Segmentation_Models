import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Reshape
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB2


def SqueezeAndExcitation(inputs, ratio=8):
    
    b, h, w, c = inputs.shape
    
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation='relu', use_bias=False)(x)
    x = Dense(c, activation='sigmoid', use_bias=False)(x)
    
    x = Multiply()([inputs, x])
    
    return x

def Pyramid_Pooling_Module(features, f=32, p1=2, p2=3, p3=8):

    shape = features.shape
    red = GlobalAveragePooling2D()(features)
    red = Reshape((1,1,shape[-1]))(red)
    red = Conv2D(filters=f, kernel_size=(1,1), padding='same', use_bias=False)(red)
    red = BatchNormalization()(red)
    red = Activation('relu')(red)
    red = UpSampling2D(size=shape[1],interpolation='bilinear')(red)
    
    orange = AveragePooling2D(pool_size=(p1))(features)
    orange = Conv2D(filters=f, kernel_size=(1,1), padding='same', use_bias=False)(orange)
    orange = BatchNormalization()(orange)
    orange = Activation('relu')(orange)
    orange = UpSampling2D(size=p1,interpolation='bilinear')(orange)

    blue = AveragePooling2D(pool_size=(p2))(features)
    blue = Conv2D(filters=f, kernel_size=(1,1), padding='same', use_bias=False)(blue)
    blue = BatchNormalization()(blue)
    blue = Activation('relu')(blue)
    blue = UpSampling2D(size=p2,interpolation='bilinear')(blue)

    green = AveragePooling2D(pool_size=(p3))(features)
    green = Conv2D(filters=f, kernel_size=(1,1), padding='same', use_bias=False)(green)
    green = BatchNormalization()(green)
    green = Activation('relu')(green)
    green = UpSampling2D(size=p3,interpolation='bilinear')(green)

    return Concatenate()([features, red, orange, blue, green])

def ASPP(image_features):

    shape = image_features.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(image_features)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=32, kernel_size=1, padding='same', dilation_rate = 12,use_bias=False)(image_features)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y_c = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(y_c)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    return y


def DeepLabV3Plus_PSPNet_SE(inputs, classes=1):

    inputs = Input(inputs)

    base_model = EfficientNetB2(weights='imagenet', include_top=False, input_tensor=inputs, drop_connect_rate=0.25)
    high_level_image_features = base_model.get_layer('block6a_expand_bn').output
    # high_level_image_features = SqueezeAndExcitation(high_level_image_features, ratio=16)
    
    x_a = ASPP(high_level_image_features)
    x_a = SqueezeAndExcitation(x_a, ratio=16)
 
    x_b = Pyramid_Pooling_Module(high_level_image_features, f=32, p1=2, p2=4, p3=8)
    x_b = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    x_b = SqueezeAndExcitation(x_b, ratio=16)

    x_c = Concatenate()([x_a, x_b])
    x_c = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = Activation('relu')(x_c)
    x_c = SqueezeAndExcitation(x_c, ratio=16)
    x_c = UpSampling2D(size=4, interpolation='bilinear')(x_c)

    low_level_image_features = base_model.get_layer('block3a_expand_bn').output
    x_d = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(low_level_image_features)
    x_d = BatchNormalization()(x_d)
    x_d = Activation('relu')(x_d)

    x_e = Concatenate()([x_c, x_d])
    x_e = Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(x_e)
    x_e = BatchNormalization()(x_e)
    x_e = Activation('relu')(x_e)
    x_e = SqueezeAndExcitation(x_e, ratio=16)

    x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(x_e)
    x = BatchNormalization(name=f'bn_8')(x)
    x = Activation('relu', name=f'relu_8')(x)

    x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_9')(x)
    x = Activation('relu', name=f'relu_9')(x)

    x = UpSampling2D(size=4, interpolation='bilinear')(x)

    """ Outputs """
    x = Conv2D(filters=classes, kernel_size=1, name='output_layer')(x)
    
    if classes == 1:
        x = Activation('sigmoid')(x)
    else:
        x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model
  
  
def main():

    model = DeepLabV3Plus_PSPNet_SE(inputs=(1024,1024,3), classes=4)
    # model.summary()


if __name__== '__main__':

    main()
