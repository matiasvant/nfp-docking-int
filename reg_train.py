import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from math import ceil
from features import \
    num_atom_features, \
    num_bond_features
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import sys
from networkP import dockingProtocol
from util import buildFeats, dockingDataset, labelsToDF
import time
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-dropout','--df',required=True)
parser.add_argument('-learn_rate','--lr',required=True)
parser.add_argument('-os','--os',required=True)
parser.add_argument('-data', '--dat', required=True)
parser.add_argument('-bs', '--batch_size', required=True)
parser.add_argument('-fplen', '--fplength', required=True)
parser.add_argument('-mnum', '--model_number', required=True)
parser.add_argument('-wd', '--weight_decay', required=True)

cmdlArgs = parser.parse_args()
df=float(cmdlArgs.df)
lr=float(cmdlArgs.lr)
oss=int(cmdlArgs.os)
wd=float(cmdlArgs.weight_decay)
bs=int(cmdlArgs.batch_size)
data = cmdlArgs.dat
fplCmd = int(cmdlArgs.fplength)
mn = cmdlArgs.model_number

print(f'interop threads: {torch.get_num_interop_threads()}, intraop threads: {torch.get_num_threads()}')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.closs = 0
        self.ccounter = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif np.abs([self.min_validation_loss - validation_loss])[0] <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def early_cstop(self, train_loss):
        if train_loss == self.closs:
            self.ccounter += 1
        else:
            self.closs = train_loss
            self.ccounter = 0
        if self.ccounter == 200:
            return True
        return False


# 70-15-15 split
smileData = pd.read_csv('../../data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smiles', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

data_path = f'../../data/{data}.txt'
allData = labelsToDF(data_path)
if 'smiles' not in allData.columns:
    allData = pd.merge(allData, smileData, on='zinc_id')

trainData, valData, testData = np.split(allData.sample(frac=1), 
                                        [int(.70*len(allData)), int(.85*len(allData))])

print(f'merged df shapes: {trainData.shape}, {valData.shape}, {testData.shape}')


ID_column = allData.columns[allData.columns.str.contains('zinc_id|Compound ID', case=False, regex=True)].tolist()[0]

xTrain = trainData[[ID_column, 'smiles']].values.tolist()
yTrain = trainData['labels'].values
xTest = testData[[ID_column, 'smiles']].values.tolist()
yTest = testData['labels'].values
xValid = valData[[ID_column, 'smiles']].values.tolist()
yValid = valData['labels'].values

print(f"yTrain: {yTrain[:5]}")
print(f"yTest: {yTest[:5]}")

scaler = StandardScaler()
yTrain = yTrain.reshape(-1, 1)
print(type(yTrain))
yTrain = scaler.fit_transform(yTrain).T[0].tolist()  
yTest = scaler.transform(yTest.reshape(-1, 1)).T[0].tolist()  # reuse scaling from train data to avoid data leakage
yValid = scaler.transform(yValid.reshape(-1, 1)).T[0].tolist()

print(f"scaled yTrain: {yTrain[:5]}")
print(f"scaled yTest: {yTest[:5]}")


trainds = dockingDataset(train=xTrain, 
                        labels=yTrain,
                        name='train')
traindl = DataLoader(trainds, batch_size=bs, shuffle=True)
testds = dockingDataset(train=xTest,
                        labels=yTest,
                        name='test')
testdl = DataLoader(testds, batch_size=bs, shuffle=True)
validds = dockingDataset(train=xValid,
                         labels=yValid,
                         name='valid')
validdl = DataLoader(validds, batch_size=bs, shuffle=True)



fpl = fplCmd 
hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
layers = [num_atom_features()] + hiddenfeats
modelParams = {
    "fpl": fpl,
    "conv": {
        "layers": layers
    },
    "ann": {
        "layers": layers,
        "ba": [fpl, fpl // 4, 1],
        "dropout": df
    }
}
print(f'layers: {layers}, through-shape: {list(zip(layers[:-1], layers[1:]))}')



model = dockingProtocol(modelParams).to(device=device)
print(model)
# print("inital grad check")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total trainable params: {totalParams}')
lossFn = nn.MSELoss() # gives 'mean' val by default
# adam, lr=0.01, weight_decay=0.001, prop=0.2, dropout=0.2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
model.load_state_dict(torch.load('../basisModel.pth'), strict=False)
lendl = len(trainds)
num_batches = len(traindl)
print("Num batches:", num_batches)
bestVLoss = 100000000
lastEpoch = False
epochs = 20  # 200 initially 
earlyStop = EarlyStopper(patience=10, min_delta=0.01)
trainLoss, validLoss = [], []
trainR, validR = [], []
for epoch in range(1, epochs + 1):
    print(f'\nEpoch {epoch}\n------------------------------------------------')
    
    stime = time.time()
    model.train()
    runningLoss, r_squared, r_list = 0, 0, []

    for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
        at, bo, ed, scaled_Y = a.to(device), b.to(device), e.to(device), y.to(device)

        scaled_preds = model((at, bo, ed))
        loss = lossFn(scaled_preds, scaled_Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        runningLoss += scaled_preds.shape[0] * loss.item()
 
        preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        Y = scaler.inverse_transform(scaled_Y.detach().cpu().numpy().reshape(-1, 1)).squeeze()
        print(f"P: {preds[:5]}, Y: {Y[:5]}")

        _,_,r_value,_,_ = linregress(preds, Y)
        r_list.append(r_value ** 2)

        cStop = earlyStop.early_cstop(loss.item())
        if cStop: break
   
        if batch % (np.ceil(lendl / bs / 10)) == 0:
            lossDisplay, currentDisplay = loss.item(), (batch + 1)
            print(f'loss: {lossDisplay:>7f} [{((batch + 1) * len(a)):>5d}/{lendl:>5d}]')

    trainLoss.append(runningLoss/lendl)
    if len(r_list) != 0:
        r_squared = sum(r_list)/len(r_list)
    trainR.append(r_squared/num_batches)
    if cStop: break
    print(f'Time to complete epoch: {time.time() - stime}')
    print(f'\nTraining Epoch {epoch} Results:\nloss: {runningLoss/lendl:>8f}, R^2: {r_squared/num_batches:>8f}\n------------------------------------------------')
    
    size = len(validdl.dataset)
    num_batches = len(validdl)
    model.eval()
    valid_loss = 0
    runningLoss, r_squared, r_list = 0, 0, []
    with torch.no_grad():
        for (a, b, e, (scaled_y, zidValid)) in validdl:
            scaled_preds = model((a, b, e))
            valid_loss += lossFn(scaled_preds.to(device), scaled_y.to(device)).item()

            preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
            y = scaler.inverse_transform(scaled_y.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
            _, _, r_value, _, _ = linregress(preds, y)
            r_list.append(r_value ** 2)
    valid_loss /= num_batches
    if len(r_list) != 0:
        r_squared = sum(r_list)/len(r_list)
    validLoss.append(valid_loss)
    validR.append(r_squared)
    print(f'\nValidation Results:\nLoss: {valid_loss:>8f}, R^2: {r_squared:>0.1f}%\n------------------------------------------------')
    
    # if valid_loss < bestVLoss:
    #     bestVLoss = valid_loss
    #     model_path = f'model_{epoch}'
    #     torch.save(model.state_dict(), model_path)

    if earlyStop.early_stop(valid_loss):
        print(f'validation loss converged to ~{valid_loss}')
        break

if cStop: 
    print(f'training loss converged erroneously')
    sys.exit(0)

epochR = range(1, epochs + 1)
plt.plot(epochR, trainLoss, label='Training Loss', linestyle='-', color='lightgreen')
plt.plot(epochR, validLoss, label='Validation Loss', linestyle='-', color='darkgreen')
plt.plot(epochR, trainR, label='Training R^2', linestyle='--', color='lightblue')
plt.plot(epochR, validR, label='Validation R^2', linestyle='--', color='darkblue')

plt.title('Training and Validation Loss/R^2')
plt.xlabel('Epochs')
plt.ylabel('Loss / R^2')
 
plt.legend(loc='best')

plt.xticks(np.arange(0, epochs + 1, 2))
 
plt.legend(loc='best')
plt.savefig(f'./loss{mn}.png')
plt.show()
plt.close()
with open(f'./lossData{mn}.txt', 'w+') as f:
    f.write('train loss, validation loss\n')
    f.write(f'{",".join([str(x) for x in trainLoss])}')
    f.write(f'{",".join([str(x) for x in validLoss])}')