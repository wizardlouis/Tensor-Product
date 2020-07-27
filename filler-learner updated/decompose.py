import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math

from trainTPE import *
from FillerTPE import FillerTPE


if __name__=="__main__":
    path='make_data//example//'
    savepath='make_data//TPE//'
    select='data_0.001'

    Symbol_Embedding=nn.Embedding(6,6,_weight=torch.eye(6))
    role_Embedding=nn.Embedding(3,3,_weight=torch.eye(3))
    TPE=FillerTPE(Symbol_Embedding,role_Embedding,free_dim=3,final_layer_width=200,n_roles=3,n_fillers=6,filler_dim=2,Symbol_learning=False,
                  binder='tpr',hidden_dim=20,bidirectional=True,num_layers=1,softmax_fillers=True)

    data=torch.load(path+select)
    Seq=data['Seq'];Vectors=data['Vectors']
    print(Seq.shape,Vectors.shape)
    Test_Seq=Seq[2::3];Test_Vectors=Vectors[2::3]
    Train_Seq=torch.cat((Seq[::3],Seq[1::3]),dim=0)
    Train_Vectors=torch.cat((Vectors[::3],Vectors[1::3]),dim=0)

    Train_data=[Train_Seq,Train_Vectors]
    Test_data=[Test_Seq,Test_Vectors]

    Embedding_trace,report_text=trainIters_TPE(Train_data,Test_data,TPE,n_epochs=500,learning_rate=0.005,batch_size=1,patience=5,use_one_hot_temperature=True,burn_in=300,weight_file=savepath+select)
    f=open(savepath+select+'report.txt','w')
    for line in report_text:
        f.write(line)
    f.close()
    torch.save(Embedding_trace,savepath+select+'Embedding_trace')



