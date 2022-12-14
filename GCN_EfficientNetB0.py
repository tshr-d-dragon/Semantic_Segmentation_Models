import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Input, Add
from tensorflow.keras.models import Model
from keras.applications.efficientnet import EfficientNetB0


def GCN_module(features, k=5):

  y_a = Conv2D(filters=21, kernel_size=(1,k), padding='same', use_bias=False)(features)
  y_a = Conv2D(filters=21, kernel_size=(k,1), padding='same', use_bias=False)(y_a)

  y_b = Conv2D(filters=21, kernel_size=(k,1), padding='same', use_bias=False)(features)
  y_b = Conv2D(filters=21, kernel_size=(1,k), padding='same', use_bias=False)(y_b)

  y = Add()([y_a, y_b])

  return y

def BR(features):

  y = Conv2D(filters=21, kernel_size=(3,3), padding='same', use_bias=False)(features)
  y = Activation('relu')(y)
  y = Conv2D(filters=21, kernel_size=(3,3), padding='same', use_bias=False)(y)

  y = Add()([y, features])

  return y

def GCN(inputs, classes=1):

    inputs = Input(inputs)

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    res_2 = base_model.get_layer('block3a_expand_activation').output         # 4x >> block3a_expand_activation
    res_3 = base_model.get_layer('block4a_expand_activation').output         # 8x >> block4a_expand_activation 
    res_4 = base_model.get_layer('block6a_expand_activation').output         # 16x >> block6a_expand_activation
    res_5 = base_model.get_layer('top_activation').output                    # 32x >> top_activation
    
    gcn_2 = GCN_module(res_2, 7)
    gcn_3 = GCN_module(res_3, 7)
    gcn_4 = GCN_module(res_4, 7)
    gcn_5 = GCN_module(res_5, 7)

    bn_2 = BR(gcn_2)
    bn_3 = BR(gcn_3)
    bn_4 = BR(gcn_4)
    bn_5 = BR(gcn_5)

    bn_4_a = UpSampling2D(size=2, interpolation='bilinear')(bn_5)
    bn_4_a = Add()([bn_4, bn_4_a])
    bn_4_a = BR(bn_4_a)
    bn_3_a = UpSampling2D(size=2, interpolation='bilinear')(bn_4_a)
    bn_3_a = Add()([bn_3, bn_3_a])
    bn_3_a = BR(bn_3_a)
    bn_2_a = UpSampling2D(size=2, interpolation='bilinear')(bn_3_a)
    bn_2_a = Add()([bn_2, bn_2_a])
    bn_2_a = BR(bn_2_a)
    bn_1_a = UpSampling2D(size=2, interpolation='bilinear')(bn_2_a)
    bn_1_a = BR(bn_1_a)
    bn_0_a = UpSampling2D(size=2, interpolation='bilinear')(bn_1_a)
    bn_0_a = BR(bn_0_a)

    """ Outputs """
    x = Conv2D(classes, (1, 1), name='output_layer')(bn_0_a)
    
    if classes == 1:
      x = Activation('sigmoid')(x)
    else:
      x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model

  
def main():

    model = GCN(inputs=(512,512,3), classes=10)
    model.summary()


if __name__== '__main__':

    main()
