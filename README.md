# MIL-VT

Credit: S. Yu et al., "MIL-VT: Multiple Instance Learning Enhanced Vision Transformer for Fundus Image Classification," in Proc. MICCAI, 2021.
Code base borrowed from: https://github.com/greentreeys/MIL-VT

Enhancement and changes made by AML GRP 15

Please run the provided notebook on Google Colab.

### Basic Requirement:
* timm==0.3.2
* torch==1.7.0
* torchvision==0.8.1
* vit-pytorch==0.6.6
* numpy==1.19.5
* opencv-python==4.5.1.48
* pandas==1.1.5
* imgaug==0.4.0



### Pretrain Weight for MIL-VT on large fundus dataset
* Please download pretrained weight of fundus image from this link:
* https://drive.google.com/drive/folders/1YgdhA7BK6Unrs2lOflOd9rPTrwm17gdf?usp=sharing
* Store the pretrain weight in 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'
