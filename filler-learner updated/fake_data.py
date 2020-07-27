import torch
import torch.nn as nn
import math
import os

def Absolute_filler(Sequence):
    return Sequence

def make_Vectors(Sequence,filler_transform,filler_embedding,
                 noise_dim,noise_level,final_dim,projection=None):
    batch_size,n_fillers=Sequence.shape[0],Sequence.shape[1]
    filler=filler_transform(Sequence)
    if  not isinstance(filler_embedding,nn.Embedding):
        filler_embedding=nn.Embedding(filler_embedding.shape[0],filler_embedding.shape[1],_weight=filler_embedding)
    filler_embedded=filler_embedding(filler).view(batch_size,-1)
    noise_state=torch.randn(batch_size,noise_dim)
    state=torch.cat((filler_embedded,noise_state),dim=1)
    print(state.shape)
    if projection is not None:
        project_W=projection
    else:
        project_W=torch.randn(final_dim,filler_embedded.shape[1])
    print(project_W.shape)
    noise_W=noise_level*torch.randn(final_dim,noise_dim)
    print(noise_W.shape)
    W=torch.cat((project_W,noise_W),dim=1)
    Vectors=torch.einsum('ac,bc->ba',W,state)
    return Vectors



if __name__=='__main__':
    path='make_data//example//'
    if not os.path.exists(path):
        os.makedirs(path)
    pi=math.pi
    param={'noise_dim':3,'noise_level':0.001}
    theta=[0,pi/3,pi*2/3,pi,pi*4/3,pi*5/3]
    filler_embedding=torch.tensor([[math.cos(i),math.sin(i)] for i in theta])
    print(filler_embedding)
    projection=torch.randn(200,6)
    p=[0,1,2,3,4,5]
    m=[]
    for i in p:
        for j in p:
            if j!=i:
                for k in p:
                    if k!=i and k!=j:
                        m.append([i,j,k])
    Seq=torch.tensor(m)
    Vectors=make_Vectors(Seq,Absolute_filler,filler_embedding,param['noise_dim'],param['noise_level'],200,projection)
    data={}
    data.update(param)
    data.update({'theta':theta,'filler_embedding':filler_embedding,'projection':projection,'Seq':Seq,'Vectors':Vectors})
    torch.save(data,path+'data_'+str(param['noise_level']))

