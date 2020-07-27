# Functions needed for training models

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np

import sys
import os

import time
import math

import pickle

from FillerTPE import FillerTPE

use_cuda = torch.cuda.is_available()

def train_TPE(Batch,TPE,TPE_optimizer,criterion):
    Sequence_list, Vectors_list=Batch[0],Batch[1]
    Vectors_list=torch.tensor(Vectors_list)
    TPE_optimizer.zero_grad()
    mse_loss,one_hot_loss,l2_norm_loss,unique_filler_loss=0,0,0,0
    if isinstance(TPE,FillerTPE):
        TPE_output,filler_predictions=TPE(Sequence_list)
        batch_one_hot_loss,batch_l2_loss,batch_unique_loss=\
        TPE.get_regularization_loss(filler_predictions)
        one_hot_loss+=batch_one_hot_loss;l2_norm_loss+=batch_l2_loss;unique_filler_loss+=batch_unique_loss
    else:
        print('Encoder should be a TPE, given '+str(type(TPE))+'instead!!!')

    mse_loss+=criterion(TPE_output,Vectors_list.unsqueeze(0))
    loss=mse_loss+one_hot_loss+l2_norm_loss+unique_filler_loss

    loss.backward()
    TPE_optimizer.step()

    return loss.data.item(),mse_loss,one_hot_loss,l2_norm_loss,unique_filler_loss

def bishuffle(bidata):
    Sdata,Vdata=bidata[0],bidata[1]
    N=len(Sdata)
    P=list(range(N))
    shuffle(P)
    Sdata_S=torch.zeros_like(Sdata)
    Vdata_S=torch.zeros_like(Vdata)
    for i in range(N):
        Sdata_S[i]=Sdata[P[i]]
        Vdata_S[i]=Vdata[P[i]]
    return [Sdata_S,Vdata_S]

def batchify(data,batch_size):
    N=len(data[0])
    n_batch=math.ceil(N/batch_size)
    batch=[]
    for i in range(n_batch):
        batch.append([data[0][i*batch_size:(i+1)*batch_size],data[1][i*batch_size:(i+1)*batch_size]])
    return batch

def trainIters_TPE(Train_data,Test_data,TPE,n_epochs,
                   learning_rate=0.001,batch_size=5,patience=3,weight_file=None,
                   use_one_hot_temperature=True,burn_in=0):
    #weight file是模型存储的路径
    TPE_optimizer=optim.Adam(TPE.parameters(),lr=learning_rate)
    criterion=nn.MSELoss()
    pre_loss=1000000
    one_hot_temperature=0.0
    if use_one_hot_temperature:
        one_hot_temperature=1.0
    count_epochs_not_improved=0
    best_loss=pre_loss

    report_text=[]
    Embedding_trace=[]
    reached_max_temp = False
    for epoch in range(n_epochs):
        Embedding_trace.append(TPE.filler_assigner.filler_embedding.weight.clone().detach())
        print("starting training epoch: "+str(epoch)+'\n')
        if burn_in == epoch:
            print('Burn in is over, turning on regularization')
            if isinstance(TPE, FillerTPE):
                TPE.use_regularization(True)
            if burn_in == 0:
                print('Setting regularization temp to {}'.format(1))
                if isinstance(TPE, FillerTPE):
                    TPE.set_regularization_temp(1)
                reached_max_temp = True

        if epoch >= burn_in and not reached_max_temp:
            temp = float(epoch - burn_in + 1) / burn_in
            if temp <= 1:
                print('Setting regularization temp to {}'.format(temp))
                TPE.set_regularization_temp(temp)
            else:
                reached_max_temp = True
        epoch_loss=0
        epoch_mse_loss = 0;epoch_one_hot_loss = 0;epoch_l2_loss = 0;epoch_unique_filler_loss = 0
        epoch_Train=bishuffle(Train_data)
        batch_Train=batchify(epoch_Train,batch_size)

        if isinstance(TPE, FillerTPE):
            TPE.train()

        for batch in batch_Train:
            loss,batch_mse_loss,batch_one_hot_loss,batch_l2_loss,batch_unique_filler_loss=train_TPE(batch,TPE,TPE_optimizer,criterion)
            epoch_mse_loss+=batch_mse_loss;epoch_one_hot_loss+=batch_one_hot_loss;epoch_l2_loss+=batch_l2_loss;epoch_unique_filler_loss+=batch_unique_filler_loss
        epoch_loss=epoch_mse_loss+epoch_one_hot_loss+epoch_l2_loss+epoch_unique_filler_loss

        Num_train_batch=len(batch_Train[0])
        epoch_loss/=Num_train_batch;epoch_mse_loss/=Num_train_batch;epoch_one_hot_loss/=Num_train_batch;epoch_l2_loss/=Num_train_batch;epoch_unique_filler_loss/=Num_train_batch
        #report training loss
        if epoch>=burn_in:
            train_report='Average Training Loss is '+str(epoch_loss.item())+' ; '+str(epoch_mse_loss.item())+' ; '+str(epoch_one_hot_loss.item())+' ; '+str(epoch_l2_loss.item())+' ; '+str(epoch_unique_filler_loss.item())+' ; \n'
        else:
            train_report='Average Training Loss is '+str(epoch_mse_loss.item())+' ; \n'
        print(train_report)
        report_text.append(train_report)

        if isinstance(TPE, FillerTPE):
            TPE.eval()
        test_loss,test_mse_loss,test_one_hot_loss,test_l2_loss,test_unique_loss=0,0,0,0,0

        batch_Test=batchify(Test_data,batch_size)

        for batch in batch_Test:
            TPE_output, filler_predictions = TPE(batch[0])
            batch_one_hot_loss, batch_l2_loss, batch_unique_loss = \
                TPE.get_regularization_loss(filler_predictions)
            batch_mse_loss = criterion(TPE_output, batch[1].unsqueeze(0))
            test_mse_loss+=batch_mse_loss;test_one_hot_loss+=batch_one_hot_loss;test_l2_loss+=batch_l2_loss;test_unique_loss+=batch_unique_filler_loss
            test_loss+=batch_mse_loss+batch_one_hot_loss+batch_l2_loss+batch_unique_filler_loss

        if reached_max_temp or burn_in == epoch:
            if test_loss < best_loss:
                print('Saving model at epoch {}'.format(epoch))
                count_epochs_not_improved = 0
                best_loss = test_loss
                torch.save(TPE,weight_file+'TPE.pth')
                torch.save(TPE.filler_assigner,weight_file+'assigner.pth')
            else:
                count_epochs_not_improved += 1
                if count_epochs_not_improved == patience:
                    print('Finished training early')
                    break

        Num_test_batch=len(batch_Test[0])
        test_loss/=Num_test_batch;test_mse_loss/=Num_test_batch;test_one_hot_loss/=Num_test_batch;test_l2_loss/=Num_test_batch;test_unique_loss/=Num_test_batch
        if epoch>=burn_in:
            test_report='Average Test Loss is '+str(test_loss.item())+' ; '+str(test_mse_loss.item())+' ; '+str(test_one_hot_loss.item())+' ; '+str(test_l2_loss.item())+' ; '+str(test_unique_loss.item())+' ; \n\n'
        else:
            test_report='Average Test Loss is '+str(test_mse_loss.item())+' ; \n'
        print(test_report)
        report_text.append(test_report)


    return Embedding_trace,report_text

def test_TPE(Test_data,TPE,weight_file=None):
    if not isinstance(TPE,FillerTPE):
        print('Expected model of FillerTPE, Given '+str(type(TPE))+' instead!!!')
    else:
        Test_Seq,Test_Vec=Test_data[0],Test_data[1]
        Test_Vec=torch.tensor(Test_Vec)
        TPE.eval()
        output,filler_prediction=TPE(Test_Seq)
        criterion=nn.MSELoss()
        mseloss=criterion(output,Test_Vec)
        return mseloss/len(Test_data[0])


