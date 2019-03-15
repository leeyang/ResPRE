#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:30:45 2017

@author: lee
"""

import sys,os
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import resnet
import precision
torch.manual_seed(6)
import random
random.seed(6)

print('cuda is ready? :',torch.cuda.is_available())
def outpre_(pre):
    [n,d]=pre.shape
    ilist=[]
    jlist=[]
    newpre=[]
    for i in range(n):
        for j in range(i+1,d):
            
            newpre.append(-pre[i,j])
            
            ilist.append(i)
            jlist.append(j)
    index=np.argsort(newpre)
    res=[]
    l=len(index)
    for i in range(l):
        res.append(str(ilist[index[i]]+1)+' '+str(jlist[index[i]]+1)+' '+str(pre[ilist[index[i]],jlist[index[i]]])+'\n')
    return res

def getweights_out(msafile,seq_id,outfile):
    exefile=os.path.join(os.path.dirname(__file__),'bin/calNf_ly')
    cmd=exefile+' '+msafile+' '+str(seq_id)+' >'+outfile
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True)
    output,error = p.communicate()

def test(model,pre):
    model.eval()
    x=torch.FloatTensor(pre)
    x=torch.unsqueeze(x,0)
    x=Variable(x,volatile=True)
    output=model(x)
    predicted=output.cpu().data. numpy()[0][0]
    

    return predicted            
      
def outpre(predicted,outfile):
    outlines=outpre_(predicted)
    woutfile=open(outfile,'w')
    for aline in outlines:
        woutfile.write(aline)
    woutfile.close()
        
            
    
if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("deeppre.py [msa_file] [outfile]")
        print("\tmsa_file: Multiple Sequence Alignment file")
        print("\toutfile: contact prediction in NeBcon format")
        exit()

    #step1: generate weights
    #step2: generate precision matrix
    #step3: prediction using deepPRE
    msafile=sys.argv[1]
    savefile=sys.argv[2]
    model_file=os.path.join(os.path.dirname(__file__),'saved_model')
    #config
    
    
    seq_id=0.8
    weightfile=savefile+'.weight'
    getweights_out(msafile,seq_id,weightfile)
    #step1 finished
    pre=precision.computepre(msafile,weightfile)
    #step2 finished
    model=resnet.resnet46()
    model.cpu()
    predicts=[]
    for i in range(10):
        strbar = '>'*(i*10//2)+' '*((100-i*10-1)//2)
        sys.stdout.write('\r'+strbar+'[%s%%]'%((i+1)*10))
        sys.stdout.flush()
        amodel_file=os.path.join(model_file,str(i+1))
        model.load_state_dict(torch.load(amodel_file, map_location=lambda storage, loc: storage))
        predicts.append(test(model,pre))
    predict=np.mean(predicts,axis=0)
    outpre(predict,savefile)
    sys.stdout.write('\ndone.')
    
    
        
