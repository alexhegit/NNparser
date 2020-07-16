# -*- coding: utf-8 -*-
import utils.pytools as pt

# Tested Models
# 1. torchvision: 
#    alexnet, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg_13 bn, vgg16_bn,
#   vgg19_bn, resnet18, resnet34, resnet50, resnet101, resnet152, 
#   SqueezeNet1_0, SqueezeNet1_1, densenet_121, densenet_169, densenet_201
#   densenet_161, inception_v3, googlenet, shufflenet_V2_x'n'_'n', mobileNet_v2
#   resnext_50_32x4d, resnext_101_32x8d, wide_resNet_50_2, wide_resnet_101_2
#   MNASNet'n'_'n'

# 2. Recomendation:
#    DLRM


# model to be loaded
nnname = "resnet50"
# producing the table of the model paramter list
(ms, depth, isconv) = pt.modelLst(nnname)    
ms = pt.tableGen(ms,depth,isconv)

# exporting the table at //output//torch//
pt.tableExport(ms,nnname)