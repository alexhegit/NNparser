# -*- coding: utf-8 -*-
# module version: convert keras model into table
# module: conversion: typical layers to rows
# convert to csv/excel
# typical models
# special layers

# to do: complete models, includeing NCF
# models in tfhub
# more outputs by formula?

import csv
import utils.tftools as tt
import utils.formattable as ft
# model tobe loaded
nnname = 'bert'

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
        training = False
        if training:
            model = get_model(token_num=len(token_dict),training=training)
        else:
            # Revise lib\site-packages\keras_bert\bert.py: line164
            # "return inputs, transformed" -> "return inputs, transformed,model"
            _,_,model = get_model(token_num=len(token_dict),training=training)
         
        compile_model(model)
    return model,isconv


(model,isconv) = GetModel(nnname) 
paralist=tt.headgen(isconv)

for x in model.layers: #model.layers[::-1]
     # no batch, hxwxc
    inp0=['']*3; inp1=['']*3; out=['']*3
    kh = ''; kw=''; sh = ''; sw=''; ph = ''; pw=''
    extin=''
    datai=''; datao=''; dataw=''
    gemm=''; vect='' ; acti =''
    ltype = str(type(x)).split(".")[-1].split("'")[0]
    # if x.name=='Encoder-1-FeedForward':
    #     print(x.name)
    #print(x.name)
    
    # input tensor & size
    (inp0, inp1, datai, extin) = tt.inputgen(x,inp0,inp1,extin)
    # output tensor & size
    (out,datao,extin) = tt.outputgen(x,out,extin)
    # weight size
    dataw = tt.getweightsize(x,dataw)
    # # of ops: gemm, elememwise, activiation(transcendental functions)
    (gemm, vect, acti) = tt.opscomputation(x,datao,inp0)
    # conv tensor
    (kh, kw, sh, sw, ph, pw) = tt.pararetrival(x)
   
    xtype=str(type(x))
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
        doublerow = False
        dim=2
        if isinstance(gemm,list): 
            if doublerow: # multihead attention: tow rows
                new_row = [x.name,ltype]+ inp0[:dim]+inp1[:dim]+out[:dim]+[datai,datao,dataw,gemm[0],vect[0],acti[0],extin]
                paralist.append(new_row)
                new_row = ['']*11+[gemm[1],vect[1],acti[1]]+['']
                paralist.append(new_row)
            else:
                new_row = [x.name,ltype]+ inp0[:dim]+inp1[:dim]+out[:dim]+[datai,datao,dataw,gemm[1],vect[1],acti[1],extin]
                paralist.append(new_row)
        else:
            new_row = [x.name,ltype]+ inp0[:dim]+inp1[:dim]+out[:dim]+[datai,datao,dataw,gemm,vect,acti,extin]
            paralist.append(new_row)

           
import  pandas as pd
df = pd.DataFrame(paralist)
paraout = './/outputs//tf//'+nnname+'.xlsx'  
with pd.ExcelWriter(paraout) as writer:
    df.to_excel(writer,sheet_name='details')
    # dfsum.to_excel(writer,sheet_name='summary',index=Flase)
    writer.save()
writer.close()

maxVal=ft.SumTable(paraout)
ft.FormatTable(paraout,maxVal)