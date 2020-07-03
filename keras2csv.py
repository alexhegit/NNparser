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
import numpy as np
import csv

# model tobe loaded
nnname = 'ncf'

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
    
    elif nnname == 'bert':
        isconv =False
        from keras_bert import get_base_dict, get_model, compile_model
        # Build token dictionary
        token_dict = get_base_dict()  
        # Build & train the model
        model = get_model(
            token_num=len(token_dict),
            head_num=5,
            transformer_num=12,
            embed_dim=25,
            feed_forward_dim=100,
            seq_len=20,
            pos_num=20,
            dropout_rate=0.05,
        )
        compile_model(model)
        #model.summary()

    return model,isconv

(model,isconv) = GetModel(nnname) 
paralist = []
# to do: adjust the names according to models
if isconv:     
    dim =3 # 4 dim tensor: BHWC, no B
    linput=['I0_'+str(i) for i in range(dim)] + ['I1_'+str(i) for i in range(dim)]
    loutput=['O_'+str(i) for i in range(dim)]
    lweights = ['K_1','K_2','S_1','S_2','p_1','p_2']
    largs =['size','','','ops','','']
    row0 = ['layer','type'] +linput + loutput + lweights + largs+['Misc']
    largs =['input','output','weight','gemm','vect','acti']
    row1 = ['','']+['']*len(linput + loutput + lweights) + largs +['']
else:
    dim=2 # 3 dim: B+ 1XW vector,no B
    linput=['I0_'+str(i) for i in range(dim)] + ['I1_'+str(i) for i in range(dim)]
    loutput=['O_'+str(i) for i in range(dim)]
    lweights = []
    largs =['size','','','ops','','']
    row0 = ['layer','type'] +linput + loutput + lweights + largs+['Misc']
    largs =['input','output','weight','gemm','vect','acti']
    row1 = ['','']+['']*len(linput + loutput + lweights) + largs +['']
paralist.append(row0)
paralist.append(row1)

for x in model.layers: #model.layers[::-1]
    out=['']*3 # no batch, hxwxc
    inp0=['']*3
    inp1=['']*3
    kh = ''; kw=''
    sh = ''; sw=''
    ph = ''; pw=''
    extin=''
    datai=''; datao=''; dataw=''
    gemm=''; vect='' ; acti =''
    ltype = str(type(x)).split(".")[-1].split("'")[0]
    if x.name=='NSP-Dense':
        print(x.name)
    #print(x.name)
    conf = x.get_config()    
    
    # input tensors
    if not isinstance(x.input, list): # single input
        datai0=1
        for i in range(1,4,1):
            try:
                inp0[i-1]=x.input.shape[i]                
            except IndexError:
                None
        for item in inp0:
            if isinstance(item,int):
                datai0=datai0*item            
        datai=(datai0)
    elif len(x.input)>1:       # 2 inputs
        datai0=1
        for i in range(1,4,1):            
            try:
                inp0[i-1]=x.input[0].shape[i]
            except IndexError:
                None
        for item in inp0:
            if isinstance(item,int):
                datai0=datai0*item 
        datai1=1
        for i in range(1,4,1):
            try:
                inp1[i-1]=x.input[1].shape[i]
            except IndexError:
                None
        for item in inp1:
            if isinstance(item,int):
                datai1=datai1*item 
        datai=(datai0+datai1)
        if len(x.input)>2:
            for inp in x.input_shape[2:]:
                tmp = inp[1:]
                dtmp = 1
                for item in tmp:
                    if isinstance(item,int):
                        dtmp=dtmp*item
                datai += dtmp
                extin = extin + str(tmp) + ', '
            extin = extin[:-1]
                
    # output: 
    if not isinstance(x.output, list): 
        # single output：2d vector or 4d tensor: batch x oh x ow x oc
        datao=1
        for i in range(1,4,1):            
            try:
                out[i-1]=x.output.shape[i]
            except IndexError:
                None
        for item in out:
            if isinstance(item,int):
                datao=datao*item     
    else: # output0,size of more outputs
        datao=1
        for i in range(1,4,1):            
            try:
                out[i-1]=x.output[0].shape[i]
            except IndexError:
                None
        for item in out:
            if isinstance(item,int):
                datao=datao*item 
        if len(x.output)>1:
            extin += ';' # addtional output
        for i in range(1,len(x.output),1):
            dtmp=1 
            tmp=x.output[i].shape
            for item in tmp:
                if isinstance(item,int):
                    dtmp=dtmp*item
            datao +=dtmp
            extin += str(tmp) + ','
        extin = extin[:-1]
    
    weights=x.get_weights()
    if len(weights)>0: 
        dataw=0
        for item in weights:
            dataw += np.prod(item.shape)
                
    xtype=str(type(x))
    if ltype:
        if ltype=='BatchNormalization': # BN
            vect = datao*2 #1 elem* 1elem+
        if ltype=='Add': #add layer
            vect = datao # output tensor size
        if ltype=='LayerNormalization': #add layer
            acti = datao # output tensor size
        if ltype=='MultiHeadAttention': #attentio layer
            head = x.head_num
            seqlen,emblen = inp0[:2]
            keylen =emblen//head # feature_dim
            gemm=0; vect=0; acti=0
            #one head of one sentence
            ub = 1 if x.use_bias else 0
            # Q*K'= X*W_q*W_k'*X'
            QS = seqlen*emblen*keylen + seqlen*keylen*ub # size: seqlen*keylen
            KS = seqlen*emblen*keylen + seqlen*keylen*ub # size: seqlen*keylen
            gemm += QS+KS + seqlen*keylen*seqlen
            # / sqrt(kenlen)
            vect += seqlen*seqlen
            # softmax
            acti += seqlen*seqlen 
            # f(Q*K') * x*W_v, Wv emblen*keylen (same dim on W_q,W_k,W_v)
            gemm += seqlen*seqlen*emblen+seqlen*emblen*keylen 
            
            # multihead
            # concate
            keylen=keylen*head
            gemm = gemm*head
            vect = vect*head
            acti = acti*head
            # x*W_o
            gemm += seqlen*keylen*emblen
                      
            # outputs: features of a sentence
            datao= datao*seqlen
            
        if ltype=='LayerNormalization': #attentio layer
            seqlen,emblen = inp0[:2]
            vect=0; acti=0
            #  along last dim, emb dim
            # mean
            vect += (seqlen-1)
            # var:sum(x*x-mx)
            vect += (seqlen*3-1)
            # std: sqrt(var+epsi)
            vect += (seqlen)
            acti += (seqlen)
            # output:( x-mx)/std
            vect += (seqlen*2)
        
        if ltype=='FeedForward': #attentio layer
            seqlen,emblen = inp0[:2]
            units=x.units
            gemm=0; acti=0
            ub = 1 if x.use_bias else 0
            # w1x+b
            gemm += seqlen*emblen*units +seqlen*units*ub
            acti += seqlen*units
            # w2x+b
            gemm += seqlen*units*emblen +seqlen*emblen*ub
            acti += seqlen*emblen
        
        if ltype=='Dense':
            gemm = datai*datao#1 add 2mac
            
    
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
        gemm=kh*kw*inp0[2]*np.prod(out)
    
    if isinstance(x,keras.layers.Activation):
       acti = datao  #activation functions
   
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
        gemm=kh*kw*inp0[2]*np.prod(out)
           
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
        vect=datao*(kh*kw-1) #max op
    
    if isinstance(x,keras.layers.GlobalAveragePooling2D):
        vect=datao*(inp0[0]*inp0[1]-1) #add op
        
    if xtype.find('Embedding')>0:
        # todo: inputdim??
        vocalsize = inp0[0]
        embsize = out[0]
        seqlen = inp0[0]
     
    # if isinstance(x, keras.layers.Lambda):
    #     print('')
        
    if isconv:
        new_row = [x.name,ltype]+ inp0+inp1+out+[kh,kw,sh,sw,ph,pw,datai,datao,dataw,gemm,vect,acti,extin]
        paralist.append(new_row)
    else:
        new_row = [x.name,ltype]+ inp0[:dim]+inp1[:dim]+out[:dim]+[datai,datao,dataw,gemm,vect,acti,extin]
        paralist.append(new_row)
        #paralist.append([x.name,ltype,ih0,iw0,ih1,iw1,oh,ow,oc,extin])

with open(paracsv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(paralist)
        
        
    
    

