import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB2


def ASPP(image_features):

  shape = image_features.shape
  
  y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
  y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
  y_pool = BatchNormalization(name=f'bn_1')(y_pool)
  y_pool = Activation('relu', name=f'relu_1')(y_pool)
  y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

  y_1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(image_features)
  y_1 = BatchNormalization(name=f'bn_2')(y_1)
  y_1 = Activation('relu', name=f'relu_2')(y_1)

  y_6 = Conv2D(filters=256, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
  y_6 = BatchNormalization(name=f'bn_3')(y_6)
  y_6 = Activation('relu', name=f'relu_3')(y_6)

  y_12 = Conv2D(filters=256, kernel_size=1, padding='same', dilation_rate = 12,use_bias=False)(image_features)
  y_12 = BatchNormalization(name=f'bn_4')(y_12)
  y_12 = Activation('relu', name=f'relu_4')(y_12)

  y_18 = Conv2D(filters=256, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
  y_18 = BatchNormalization(name=f'bn_5')(y_18)
  y_18 = Activation('relu', name=f'relu_5')(y_18)

  y_c = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

  y = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_c)
  y = BatchNormalization(name=f'bn_6')(y)
  y = Activation('relu', name=f'relu_6')(y)

  return y


def DeepLabV3Plus(inputs, classes=1):

    inputs = Input(inputs)

    base_model = EfficientNetB2(weights='imagenet', include_top=False, input_tensor=inputs)
    high_level_image_features = base_model.get_layer('block6a_expand_bn').output

    x_a = ASPP(high_level_image_features)
    x_a = UpSampling2D(size=4, interpolation='bilinear')(x_a)

    low_level_image_features = base_model.get_layer('block3a_expand_bn').output

    x_b = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(low_level_image_features)
    x_b = BatchNormalization(name=f'bn_7')(x_b)
    x_b = Activation('relu', name=f'relu_7')(x_b)

    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_8')(x)
    x = Activation('relu', name=f'relu_8')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_9')(x)
    x = Activation('relu', name=f'relu_9')(x)

    x = UpSampling2D(size=4, interpolation='bilinear')(x)

    """ Outputs """
    x = Conv2D(classes, (1, 1), name='output_layer')(x)
    
    if classes == 1:
      x = Activation('sigmoid')(x)
    else:
      x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model

  
def main():

    model = DeepLabV3Plus(inputs=(1024,1024,3), classes=4)
    model.summary()


if __name__== '__main__':

    main()
