import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, Concatenate, Input, Reshape
from keras.models import Model
from keras.applications.resnet import ResNet50


def Pyramid_Pooling_Module(features, f=64, p1=2, p2=3, p3=6):

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


def PSPNet(inputs, classes=100):

    inputs = Input(inputs)

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model_features = base_model.get_layer('conv3_block4_add').output

    x = Pyramid_Pooling_Module(base_model_features, f=64, p1=2, p2=4, p3=8)
    x = UpSampling2D(size=8, interpolation='bilinear')(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    """ Outputs """
    x = Conv2D(classes, (1, 1), name='output_layer')(x)
    print(x.shape)

    if classes == 1:
      x = Activation('sigmoid')(x)
    else:
      x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model

  
def main():

    model = PSPNet(inputs=(256,256,3), classes=100)
    # model.summary()


if __name__== '__main__':

    main()
