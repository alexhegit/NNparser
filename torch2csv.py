# -*- coding: utf-8 -*-
import torch
from utils.torchsummary import summary
from torchvision import models

# torchvision: 
#    alexnet, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg_13 bn, vgg16_bn,
#   vgg19_bn, resnet18, resnet34, resnet50, resnet101, resnet152, 
#   SqueezeNet1_0, SqueezeNet1_1, densenet_121, densenet_169, densenet_201
#   densenet_161, inception_v3, googlenet, shufflenet_V2_x'n'_'n', mobileNet_v2
#   resnext_50_32x4d, resnext_101_32x8d, wide_resNet_50_2, wide_resnet_101_2
#   MNASNet'n'_'n'
# Transfomer: 
#    bert-base-cased
# Recomendation:
#    DLRM


# model to be loaded
nnname = "resnet18"#"bert-base-cased"

isconv = True
depth = 4

#two type of inputs in () 
# real instances： real example in tensor format: () for multi-data,[] for args
# shapes: tuple of shape, to produce random-value inputs; () for data,[] for args; 
# vision models in torchvision
if hasattr(models,nnname):
    model = getattr(models, nnname)()
    shape=(3, 224, 224)
    ms=str(summary(model,shape, depth=depth,branching=2,verbose=1))

if nnname =='dlrm':
    isconv = False
    from torchmodels.dlrm.dlrm_s_pytorch import model
    model=model()    # model(x,lS_O,lS_i)
    #Pseudo input
    x = torch.rand(2,4) # dual samples
    lS_o = torch.Tensor([[0,1],[0,1],[0,1]]).to(torch.long)
    lS_i = [torch.Tensor([1,0,1]).to(torch.long),torch.Tensor([0,1]).to(torch.long),torch.Tensor([1,0]).to(torch.long)] 
    inst = (x,[lS_o,lS_i])
    if isconv:
        ms=str(summary(model,inst, depth=depth,branching=2,verbose=1,device='cpu'))
    else:
        col_names =("input_size","output_size", "num_params")
        ms=str(summary(model,inst, col_names=col_names, depth=depth,branching=2,verbose=1,device='cpu'))

if nnname =='bert-base-cased':
    isconv = False
    from transformers import AutoModel # using Huggingface's version
    model = AutoModel.from_pretrained(nnname)
    # psudeo input
    inst = torch.randint(100,2000,(1,7))
    
    depth = 2
    if isconv:
        ms=str(summary(model,inst, depth=depth,branching=2,verbose=1))
    else:
        col_names =("input_size","output_size", "num_params")
        ms=str(summary(model,inst, col_names=col_names,depth=depth,branching=2,verbose=1))
    
# csv gen
header = 'layer' + ','*(depth)
if isconv:
    header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
    header += 'O1,O2,O3,' # output: coxhxw
    header += 'k1,k2,' # kernel
    header += 's1,s2,' # stride
    header += 'p1,p2,' # padding
    header += '#Para' # # of parameters
    header += '\n'
else: # FC style networks
    header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
    header += 'O1,O2,O3,' # output: coxhxw
    header += '#Para' # # of parameters
    header += '\n'
ms = header + ms
fname=".//outputs//torch//" +nnname+".csv"
with open(fname,"w") as f:
        f.write(ms)