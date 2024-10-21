import os
import argparse
import pandas as pd
from networkP import dockingProtocol
from features import num_atom_features
from torch import save
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True)
parser.add_argument('-fpl', '--fplength', required=True)

ioargs = parser.parse_args()
data = ioargs.data
fplength = ioargs.fplength

# dropout = [0.3, 0.5, 0.7]
# learn_rate = [0.001, 0.0001, 0.003, 0.0003]
# weight_decay = [0.0001, 0, 0.001]
# oss = [25]
# bs = [64, 128, 256]

dropout = [0.0]                # static, for testing
learn_rate = [0.001, 0.0001, 0.0003]
weight_decay = [0.0001, 0]
oss = [25]
bs = [128, 256]
fpl = [32, 64, 128]
ba = np.array([[2, 4], [2, 4, 8], [2, 8]], dtype=object)

models_hps = [
    [0, 0.001, 0.0001, 25, 64, 32],
    [0, 0.001, 0, 25, 128, 32],
    [0, 0.0001, 0.0001, 25, 256, 64],
    [0, 0.001, 0.0001, 25, 128, 32],
    [0, 0.001, 0.0001, 25, 256, 128]
]

hps = []
# for ossz in oss:
#     for batch in bs:
#         for do in dropout:
#             for lr in learn_rate:
#                 for wd in weight_decay:
#                     hps.append([ossz,batch,do,lr,wd])

for i in range(10):
    hps.append([
        25,
        np.random.choice(bs),
        0,
        np.random.choice(learn_rate),
        np.random.choice(weight_decay),
        np.random.choice(fpl),
        np.random.choice(ba)
    ])

print(f'num models: {len(hps)}')                               

try:
    os.mkdir('./src')
except:
    pass

try:
    os.mkdir('./src/trainingJobs')
except:
    pass

try:
    os.mkdir('./src/logs')
except:
    pass

with open('./src/hpResults.csv', 'w+') as f:
    f.write(f'model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,fingerprint length,validation auc,validation prauc,validation precision,validation recall,validation f1,validation hits,test auc,test prauc,test precision,test recall,test f1,test hits\n')
    

for f in os.listdir('./src/trainingJobs/'):
    os.remove(os.path.join('./src/trainingJobs', f))
for f in os.listdir('./src/logs/'):
    os.remove(os.path.join('./src/logs', f))
    
for i in range(len(hps)):
    with open(f'./src/trainingJobs/train{i + 1}.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('cd ./src/trainingJobs\n')
        f.write('module load python-libs/3.0\n')
        # f.write('source ../../tensorflow_gpu/bin/activate\n')
 
        o,batch,do,lr,wd,fpl,ba = hps[i]
        f.write('python '+'../../reg_train.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+'-data '+data+' '+'-fplen '+str(fpl)+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+' '+'-ba '+','.join(list(map(str, ba)))+'\n')


# need to update when updating model params
for i, m in enumerate(hps):
    fpl = int(m[-2]) 
    ba = m[-1]
    print(ba,  [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1])
    hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
    layers = [num_atom_features()] + hiddenfeats
    modelParams = {
        "fpl": fpl,
        "activation": 'regression',
        "conv": {
            "layers": layers
        },
        "ann": {
            "layers": layers,
            "ba": [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1],
            "dropout": 0.1 #arbitrary
        }
    }
    model = dockingProtocol(modelParams)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model trainable params: {pytorch_total_params}')
    save(model.state_dict(), f'./src/basisModel{i+1}.pth')
