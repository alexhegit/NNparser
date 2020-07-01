# -*- coding: utf-8 -*-
# module version: convert keras model into table
# module: conversion: typical layers to rows
# convert to csv/excel
# typical models
# special layers

# to do: complete models, includeing NCF
# models in tfhub
# more outputs by formula?
import tensorflow.keras as keras

import csv

# model tobe loaded
nnname = 'ResNet50'

# csv file to be exported
paracsv = './/outputs//tf//'+nnname+'.csv'


def GetModel(nnname):
    isconv = True
    # keras pretrianed models: 
    import tensorflow.keras.applications as nn
    # 'DenseNet121',  'DenseNet169',  'DenseNet201',
    # 'InceptionResNetV2',  'InceptionV3',
    # 'MobileNet',  'MobileNetV2',
    # 'NASNetLarge', 'NASNetMobile',
    # 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2',
    # 'VGG16',  'VGG19',
    # 'Xception',
    if hasattr(nn,nnname):
        model = getattr(nn, nnname)(weights=None)
        
    # efficientnet: B0-B7
    elif nnname[:-2] == 'EfficientNet':
        import tfmodels.efficientnet.tfkeras as nn
        model = getattr(nn, nnname)(weights=None)
    
    # TF2.x Models:
    elif nnname == 'ncf':
        import tfmodels.ncf as nn
        name = 'ncfmodel'
        model = getattr(nn, name)(istrain=False)
        isconv = False
    
    elif nnname == 'din':
        import tfmodels.din as nn
        name = 'din'
        _, model = getattr(nn, name)(item_count=63001, cate_count=801, hidden_units=128)
        isconv = False
    
    return model,isconv

(model,isconv) = GetModel(nnname) 
paralist = []
# to do: adjust the names according to models
if isconv:     
    dim =3 # 4 dim tensor: BHWC, no B
    linput=['I0_'+str(i) for i in range(dim)] + ['I1_'+str(i) for i in range(dim)]
    loutput=['O_'+str(i) for i in range(dim)]
    lweights = ['K_1','K_2','S_1','S_2','p_1','p_2','input','output','weight']
    heads = linput + loutput + lweights + ['Misc']
else:
    dim=2 # 3 dim: B+ 1XW vector,no B
    linput=['I0_'+str(i) for i in range(dim)] + ['I1_'+str(i) for i in range(dim)]
    loutput=['O_'+str(i) for i in range(dim)] 
    lweights = ['input','output','weight']
    heads = linput + loutput + lweights + ['Misc']
paralist.append(['layer','type'] + heads)

for x in model.layers: #model.layers[::-1]
    out=['']*3 # no batch, hxwxc
    inp0=['']*3
    inp1=['']*3
    kh = ''; kw=''
    sh = ''; sw=''
    ph = ''; pw=''
    extin=''
    datai=''
    datao=''
    dataw=''
    ltype = str(type(x)).split(".")[-1].split("'")[0]
    # if x.name == 'attention':
    #     print(x.name)
    # print(x.name)
    conf = x.get_config()    
    
    # input tensors
    if not isinstance(x.input, list): # single input
        datai0=1
        for i in range(1,4,1):
            try:
                inp0[i-1]=x.input.shape[i]                
            except IndexError:
                for ind in range(i-1):
                    if inp0[ind] is not None:
                        datai0=datai0*inp0[ind]
				break
        datai=str(datai0)
    elif len(x.input)>1:       # 2 inputs
        datai0=1
        for i in range(1,4,1):            
            try:
                inp0[i-1]=x.input[0].shape[i]
            except IndexError:
                for ind in range(i-1):
                    if inp0[ind] is not None:
                        datai0=datai0*inp0[ind]
                break
        datai1=1
        for i in range(1,4,1):
            try:
                inp1[i-1]=x.input[1].shape[i]
            except IndexError:
                for ind in range(i-1):
                    if inp1[ind] is not None:
                        datai1=datai1*inp1[ind]
                break
        datai=str(datai0+datai1)
        if len(x.input)>2:
            for inp in x.input_shape[2:]:
                tmp = inp[1:]
                extin = extin + str(tmp) + '\n'
            extin = extin[:-1]
                

    # output: 
    if not isinstance(x.output, list): 
        # single output：2d vector or 4d tensor: batch x oh x ow x oc
        datao=1
        for i in range(1,4,1):            
            try:
                out[i-1]=x.output.shape[i]
            except IndexError:
                for ind in range(i-1):
                    if out[ind] is not None:
                        datao=datao*out[ind]
                break
    else:
        print(conf['name'] + ' has ' +str(len(x.output))+' outputs')
        
    # Conv2d, MaxPooling2D, 
    if isinstance(x, keras.layers.Conv2D):
        # kernel size
        kh = conf['kernel_size'][0]
        kw = conf['kernel_size'][1]
        # stride size
        sh = conf['strides'][0]
        sw = conf['strides'][1]
        # padding
        if conf['padding']=='valid':
            ph=0
            pw=0
        elif conf['padding']=='same':
            ph=kh//2
            pw=kw//2
        dataw=inp0[2]*kh*kw*out[2]
    
    if isinstance(x, keras.layers.DepthwiseConv2D):
        # kernel size
        kh = conf['kernel_size'][0]
        kw = conf['kernel_size'][1]
        # stride size
        sh = conf['strides'][0]
        sw = conf['strides'][1]
        # padding
        if conf['padding']=='valid':
            ph=0
            pw=0
        elif conf['padding']=='same':
            ph=kh//2
            pw=kw//2
        dataw=inp0[2]*kh*kw*1
           
    if isinstance(x, keras.layers.MaxPooling2D): # ignore GlobalAveragePooling2D
        # kernel size
        kh = conf['pool_size'][0]
        kw = conf['pool_size'][1]
        # stride size
        sh = conf['strides'][0]
        sw = conf['strides'][1]
        # padding
        if conf['padding']=='valid':
            ph=0
            pw=0
        elif conf['padding']=='same':
            ph=kh//2
            pw=kw//2
    
      
    if isinstance(x, keras.layers.Embedding):
        # dim 3:
        inp0[2] = x.input_dim 
     
    # if isinstance(x, keras.layers.Lambda):
    #     print('')
        
    if isconv:
        new_row = [x.name,ltype]+ inp0+inp1+out+[kh,kw,sh,sw,ph,pw,datai,datao,dataw,extin]
        paralist.append(new_row)
    else:
        new_row = [x.name,ltype]+ inp0[:dim]+inp1[:dim]+out[:dim]+[datai,datao,dataw,extin]
        paralist.append(new_row)
        #paralist.append([x.name,ltype,ih0,iw0,ih1,iw1,oh,ow,oc,extin])

with open(paracsv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(paralist)
        
        
    
    

