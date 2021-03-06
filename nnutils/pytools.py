# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:47:13 2020

@author: LL
"""

import torch
from nnutils.torchsummary import summary
# from torch.autograd import Variable
import nnutils.formattable as ft
import nnutils.dotGen as dg
from torchvision import models
import  pandas as pd

def modelLst(ucfg):
    ''' ucfg: user's Config for the table output: nnname, BS, BPE '''
    
    nnname = ucfg['nnname']
    
    # produce config list of models per layer of the given nn model name
    isconv = True
    depth = 4
    col_names_noconv=("input_size","output_size", "num_in","num_out","num_params","gemm","vect","acti")
   

    if nnname == 'newmodel':
        import sys
        sys.path.append("..")
        from newmodel import pymodel
        x,model,depth, isconv,y = pymodel()
        if isconv:
            ms=str(summary(model,x, depth=depth,branching=2,verbose=1,ucfg=ucfg))
        else:
            ms=str(summary(model,x, col_names=col_names_noconv, depth=depth,branching=2,verbose=1,ucfg=ucfg))
        sys.path.remove("..")
        
    # vision models in torchvision
    if hasattr(models,nnname):
        model = getattr(models, nnname)()
        x = torch.rand(1,3,224,224)
        y = model(x)        
        ms=str(summary(model,x, depth=depth,branching=2,verbose=1,ucfg=ucfg))
            
    if nnname=='maskrcnn':
        depth = 6
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        model.eval()
        x = [torch.rand(3, 800,800)]
        y = model(x)
        y = y[0]
        x = [torch.rand(1,3, 800, 800)]
        ms=str(summary(model,(x,), depth=depth,branching=2,verbose=1,ucfg=ucfg))

    
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
        y = model(x,lS_o,lS_i)
        inst = (x,[lS_o,lS_i])        
        if isconv:
            ms=str(summary(model,inst, depth=depth,branching=2,verbose=1,ucfg=ucfg))
        else:
            col_names =col_names_noconv
            ms=str(summary(model,inst, col_names=col_names, depth=depth,branching=2,verbose=1,ucfg=ucfg))
    
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
            ms=str(summary(model,inst, col_names=col_names,depth=depth,branching=2,verbose=1,ucfg=ucfg))
  
    if nnname =='mymodel':
        depth=2
        isconv = False

        ## ===== To add a customized model ====
        
        # model cfgs
        N, D_in, H, D_out = 64, 1000, 100, 10
        # Create random input Tensors 
        x = torch.randn(N, D_in)
     
        # define the NN model using pytorch operators.
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )

        ## ===== end of your codes  ======

        y = model(x)
        ms=str(summary(model,x, depth=depth,branching=2,verbose=1,ucfg=ucfg))
        

    if nnname =='lstm':
        depth=2
        isconv = False
        from torchmodels.lstm import LSTMNet
        x=torch.rand(2,1).to(torch.long)
        model = LSTMNet()
        y = model(x)
        col_names =("input_size","output_size", "num_in","num_out","num_params","gemm","vect","acti")
        ms=str(summary(model,x, col_names=col_names,depth=depth,branching=2,verbose=1,ucfg=ucfg))
        
    if nnname =='gru':
        depth=2
        isconv = False
        from torchmodels.gru import GRUNet
        x=torch.rand(2,1).to(torch.long)
        model = GRUNet()
        y = model(x)
        col_names =("input_size","output_size", "num_in","num_out","num_params","gemm","vect","acti")
        ms=str(summary(model,x, col_names=col_names,depth=depth,branching=2,verbose=1,ucfg=ucfg))
        
    if nnname == 'ssd_mobilenet':
        depth = 3
        branching=2
        from torchmodels.ssd.ssd_mobilenet_v1 import create_mobilenetv1_ssd
        model = create_mobilenetv1_ssd(91)
        model.eval()
        x = torch.rand(1,3, 300, 300)
        y = model(x)
        ms=str(summary(model,(x,), depth=depth,branching=branching,verbose=1,ucfg=ucfg))
        if branching==0:
            depth=0
            
    if nnname == 'ssd_r34':
        depth = 6
        branching=2
        from torchmodels.ssd.ssd_r34 import SSD_R34
        model = SSD_R34()
        #model.eval()
        x = torch.rand(1,3,1200,1200)
        y = model(x)
        ms=str(summary(model,(x,), depth=depth,branching=branching,verbose=1,ucfg=ucfg))
        
    if nnname == 'gnmt':
        depth = 4
        isconv = False
        col_names =col_names_noconv
        from torchmodels.seq2seq.models.gnmt import GNMT
        model_config = {'hidden_size': 1024,
                    'num_layers': 4,
                    'dropout': 0.2, 'batch_first': True,
                    'share_embedding': True}
        model = GNMT(vocab_size=2048, **model_config)
        model.eval()
        seqlen=1
        batch=2
        x=torch.rand(batch,seqlen).to(torch.long)
        srclen=torch.ones(batch).to(torch.long)*seqlen
        y=model(x,srclen,x)
        ms=str(summary(model,([x,srclen,x],), col_names=col_names,depth=depth,branching=2,verbose=1,ucfg=ucfg))
        
    if nnname == 'crnn':
        depth = 6
        from torchmodels.crnn import CRNN
        model = CRNN(32, 1, 37, 256)
        x = torch.rand(1,1,32,100)
        model.eval()
        y = model(x)
        ms=str(summary(model,(x,), depth=depth,branching=2,verbose=1,ucfg=ucfg))

        
    return ms, depth, isconv,y

# table gen
def tableGen(ms,depth,isconv):
    # produce table text lst
    header=''
    if depth ==0:
        header='layer,'
    else:
        for i in range(depth):
            header += 'layer_l{},'.format(i)
        
    if isconv:
        header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
        header += 'O1,O2,O3,' # output: coxhxw
        header += 'k1,k2,' # kernel
        header += 's1,s2,' # stride
        header += 'p1,p2,' # padding
        header += 'SizeI,SizeO,SizeW,' # # of parameters
        header += 'OpGemm,OpElem,OpActi,'
        header += '\n'
    else: # FC style networks
        header += 'I1,I2,I3,' # input: cinxhxw; multiple input in model statistics
        header += 'O1,O2,O3,' # output: coxhxw
        header += 'SizeI,SizeO,SizeW,' # of parameters
        header += 'OpGemm,OpElem,OpActi,'
        header += '\n'
    ms = header + ms
    return ms

def tableExport(ms,nnname,y):        
    ms =ms.split('\n')
    ms = ms[:-1] # remove the last row 
    paralist=[]
    for row in ms:
        lst=row.split(',')
        for i in range(len(lst)):
            lst[i] = int(lst[i]) if lst[i].strip().isnumeric() else lst[i].strip()
        paralist.append(lst)
        
    df = pd.DataFrame(paralist)
    df.drop(df.columns[[-1]],axis=1,inplace = True) # remove last column
    paraout = './/outputs//torch//'+nnname+'.xlsx'  
    with pd.ExcelWriter(paraout) as writer:
        df.to_excel(writer,sheet_name='details')
        writer.save()
    writer.close()
    maxVal=ft.SumTable(paraout)
    ft.FormatTable(paraout,maxVal)
    
    if True:
        if isinstance(y,dict):
            for k,v in y.items():
                if v.grad_fn:
                    outputname ='.//outputs//torch//'+nnname+'_'+k
                    dg.graph(v,outputname)
        elif 'ssd_mo' in nnname :
            yname = ('scores','boxes' )
            for v,name in zip(y,yname):
               outputname ='.//outputs//torch//'+nnname+'_'+name
               dg.graph(v,outputname)
        elif 'ssd_r' in nnname :
            yname = ('boxes','label','scores' )
            for v,name in zip(y,yname):
                if v[0].grad_fn:
                   outputname ='.//outputs//torch//'+nnname+'_'+name
                   dg.graph(v[0],outputname)
        elif 'crnn' == nnname:
            print()  # try: except CalledProcessError:
        else: # general case, plot using the first output
            v=y[0]
            if isinstance(v[0],torch.Tensor):
                if v[0].grad_fn:
                       outputname ='.//outputs//torch//'+nnname
                       try:
                           dg.graph(v[0],outputname)
                       except :
                           print('Failed to generate model Graph')
