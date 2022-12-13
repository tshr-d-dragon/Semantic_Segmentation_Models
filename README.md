# Semantic_Segmentation_Models
I am aiming to write different Semantic Segmentation models from scratch with different pretrained backbones.

## 1.  DeepLabV3plus with SqueezeAndExcitation: 

### Paper(DeepLabV3plus): [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/pdf/1802.02611.pdf)
![Image1](https://production-media.paperswithcode.com/models/Screen_Shot_2021-02-21_at_10.34.37_AM_kvOFts0.png)
### Paper(SqueezeAndExcitation): https://arxiv.org/pdf/1709.01507.pdf
![Image2](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.55.54_PM.png)


### Implementation:
1.  [DeepLabV3plus with EfficientNet as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plus_EfficientNet.py)
2.  [DeepLabV3plus_SqueezeExcitation with EfficientNet as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plusSE_EfficientNet.py)
3.  [DeepLabV3plus with ResNet as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plus_ResNet.py)
4.  [DeepLabV3plus with DenseNet as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plus_DenseNet.py)
5.  [DeepLabV3plus with SqueezeNet as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plus_SqueezeNet.py)
6.  [DeepLabV3plus with VGG16 as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/DeepLabV3plus_VGG16.py)


## 2. Global Convolutional Network (GCN):
### Paper link: [GCN](https://arxiv.org/pdf/1703.02719.pdf)
![Image3](https://miro.medium.com/max/4800/1*4VRH-f6OaHxqyjUviJtpfg.webp)
### Implementation:
1.  [GCN with ResNet50_v2 as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/GCN_ResNet50_v2.py)


## 3.  PSPNet:

### Paper link: [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
![Image4](https://production-media.paperswithcode.com/methods/new_pspnet-eps-converted-to.jpg)
### Implementation:
1.  [PSPNet with ResNet50 as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/PSPNet_ResNet.py)


## 4.  Unet:

### Paper link: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
![Image5](https://miro.medium.com/max/1200/1*f7YOaE4TWubwaFF7Z1fzNw.png)
### Implementation:
1.  [Unet with MobileNetV2 as a backbone](https://github.com/tshr-d-dragon/Semantic_Segmentation_Models/blob/main/Unet_MobileNetV2.py)
2.  [Unet with EfficientNet as a backbone]() Coming Soon...
3.  [Unet with ResNet50 as a backbone]() Coming Soon...

**Note**: We can directly use [segmentation_models](https://segmentation-models.readthedocs.io/en/latest/) package for Unet. It reduces the number of lines of code (Github: http://github.com/qubvel/segmentation_models).
