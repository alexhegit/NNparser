# -*- coding: utf-8 -*-
import torch
from utils.torchsummary import summary
import utils.formattable as ft
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
nnname = "dlrm"#"bert-base-cased"

isconv = True
depth = 4
col_names_noconv=("input_size","output_size", "num_in","num_out","num_params","gemm","vect","acti")
#two type of inputs in () 
# real instances： real example in tensor format: () for multi-data,[] for args
# shapes: tuple of shape, to produce random-value inputs; () for data,[] for args; 
# vision models in torchvision
if hasattr(models,nnname):
    model = getattr(models, nnname)()
    shape=(3, 224, 224)
    ms=str(summary(model,shape, depth=depth,branching=2,verbose=1))

if nnname =='dlrm':
    depth=2
    isconv = False
    from torchmodels.dlrm.dlrm_s_pytorch import DLRM_Net
    import numpy as np
    # Setting for Criteo Kaggle Display Advertisement Challenge
    m_spa=16
    ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572])
    ln_bot=np.array([13,512,256,64,16])
    ln_top=np.array([367,512,256,1])
    model= DLRM_Net(m_spa,ln_emb,ln_bot,ln_top,
            arch_interaction_op="dot",
            sigmoid_top=ln_top.size - 2,
            qr_operation=None,
            qr_collisions=None,
            qr_threshold=None,
            md_threshold=None,
        )
    x = torch.rand(2,ln_bot[0]) # dual samples
    lS_i = [torch.Tensor([0,1,2]).to(torch.long)]*len(ln_emb) # numof indices >=1, but < ln_emb[i]
    lS_o = torch.Tensor([[0,2]]*len(ln_emb)).to(torch.long)
    inst = (x,[lS_o,lS_i])
    if isconv:
        ms=str(summary(model,inst, depth=depth,branching=2,verbose=1,device='cpu'))
    else:
        col_names =col_names_noconv
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
        col_names =col_names_noconv
        ms=str(summary(model,inst, col_names=col_names,depth=depth,branching=2,verbose=1))
    
# csv gen
header = 'layer' + ','*(depth)
header=''
for i in range(depth):
    header += 'layer_l{},'.format(i)
    
if isconv:
    header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
    header += 'O1,O2,O3,' # output: coxhxw
    header += 'k1,k2,' # kernel
    header += 's1,s2,' # stride
    header += 'p1,p2,' # padding
    header += 'SizeI,SizeO,SizeW,' # # of parameters
    header += 'OpGemm,OpVect,OpActi,'
    header += '\n'
else: # FC style networks
    header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
    header += 'O1,O2,O3,' # output: coxhxw
    header += 'SizeI,SizeO,SizeW,' # of parameters
    header += 'OpGemm,OpVect,OpActi,'
    header += '\n'
ms = header + ms

# fname=".//outputs//torch//" +nnname+".csv"
# with open(fname,"w") as f:
#         f.write(ms)
        
ms =ms.split('\n')
ms = ms[:-1] # remove the last row 
paralist=[]
for row in ms:
    lst=row.split(',')
    for i in range(len(lst)):
        lst[i] = int(lst[i]) if lst[i].strip().isnumeric() else lst[i].strip()
    paralist.append(lst)
    
import  pandas as pd
df = pd.DataFrame(paralist)
df.drop(df.columns[[-1]],axis=1,inplace = True) # remove last column
paraout = './/outputs//torch//'+nnname+'.xlsx'  
with pd.ExcelWriter(paraout) as writer:
    df.to_excel(writer,sheet_name='details')
    writer.save()
writer.close()

maxVal=ft.SumTable(paraout)
ft.FormatTable(paraout,maxVal)