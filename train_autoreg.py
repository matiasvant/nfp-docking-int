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
from networkP_autoreg import GCN_Autoreg
from util import *
import time
from scipy.stats import linregress
from scipy import stats
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

# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')
#         self.closs = 0
#         self.ccounter = 0

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif np.abs([self.min_validation_loss - validation_loss])[0] <= self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False
    
#     def early_cstop(self, train_loss):
#         if train_loss == self.closs:
#             self.ccounter += 1
#         else:
#             self.closs = train_loss
#             self.ccounter = 0
#         if self.ccounter == 200:
#             return True
#         return False

fpl = fplCmd 
hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
layers = [num_atom_features(just_structure=True)] + hiddenfeats
params = {
    "fpl": fpl,
    "conv": {
        "layers": layers
    },
    "mlps": {
        # "ba": [.8,.5,.5, .25],
        # "dropout": df
    }
}


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


ID_column = get_ID_type(allData)

xTrain = trainData[[ID_column, 'smiles']].values.tolist()
yTrain = trainData['labels'].values
xTest = testData[[ID_column, 'smiles']].values.tolist()
yTest = testData['labels'].values
xValid = valData[[ID_column, 'smiles']].values.tolist()
yValid = valData['labels'].values

scaler = StandardScaler()
yTrain = yTrain.reshape(-1, 1)
print(type(yTrain))
yTrain = scaler.fit_transform(yTrain).T[0].tolist()  
yTest = scaler.transform(yTest.reshape(-1, 1)).T[0].tolist()  # reuse scaling from train data to avoid data leakage
yValid = scaler.transform(yValid.reshape(-1, 1)).T[0].tolist()


trainds = dockingDataset(train=xTrain, 
                        labels=yTrain,
                        name='train', just_structure=True)
traindl = DataLoader(trainds, batch_size=bs, shuffle=True)
testds = dockingDataset(train=xTest,
                        labels=yTest,
                        name='test', just_structure=True)
testdl = DataLoader(testds, batch_size=bs, shuffle=True)
validds = dockingDataset(train=xValid,
                         labels=yValid,
                         name='valid', just_structure=True)
validdl = DataLoader(validds, batch_size=bs, shuffle=True)



model = GCN_Autoreg(params).to(device)

def remove_edges_above_node_idx(idx,a,b,e):
    # find connections to atoms beyond current specified node/subgraph
    threshold_val = idx
    replace_value = -1
    mask = e > threshold_val
    indices = torch.nonzero(mask, as_tuple=False)

    # remove the connections from edges; scrub bond features
    none_bond_features = T.tensor([1,0,0,0], dtype=torch.float)
    mol_idxs = indices[:, 0]
    atom_idxs = indices[:, 1]
    bond_idxs = indices[:, 2]

    # print(f"--Up to {i}th atom--")
    # print(f"Before", sbgr_e[int(mol_idxs[0]), int(atom_idxs[0]), :]) # picks a molecule, shows u its before n after
    # print("Edge feats:", sbgr_b[mol_idxs[0], atom_idxs[0], bond_idxs[0],:])
    b[mol_idxs, atom_idxs, bond_idxs,:] = none_bond_features
    e[mask] = replace_value
    # print("After:", sbgr_e[int(mol_idxs[0]), int(atom_idxs[0]), :])
    # print("Edge feats:", sbgr_b[mol_idxs[0], atom_idxs[0], bond_idxs[0],:]) 
    # print("new e:", sbgr_e)

    return a,b,e

def replace_elems_w_row_indices(matrix):
    """Replaces elems. w row-wise indices; except for -1 which it leaves untouched"""
    mask = matrix != -1
    row_indices = np.tile(np.arange(matrix.shape[1]), (matrix.shape[0], 1))
    result = np.where(mask, row_indices, -1)
    return result

def NLL_loss(true, mean, var):
    var = T.maximum(var, T.tensor(0.1))  # min var for numerical stability
    err = true - mean
    standardized_error = err / var

    # Standard gauss err -> the pdf density per input err (as standard gaussian)
    inner = T.exp((-T.square(standardized_error) / 2))
    constant = 1 / T.sqrt(torch.tensor(2 * torch.pi))
    pdf_value = constant * inner
    pdf_value = T.maximum(pdf_value, T.tensor(1e-10))  # numerical stability
    # ref_pdf_value = stats.norm.pdf(standardized_error) # official implementation; would require detaching gradient so not used

    prod = T.sum(pdf_value)
    err_term = -T.log(prod)

    one_over_a = T.ones_like(var) / var
    oa_prod = T.sum(one_over_a)
    var_term = -T.log(oa_prod)

    # print(f"Err: {err}")
    # print(f"1 / Var: {1 / var}")
    # print(f"Standardized err: {standardized_error}")
    # print(f"Pdf Values: {pdf_value}")
    # print(f"Prod: {prod}")
    # print("Err term:", err_term.item(), "Var term:", var_term.item())

    loss = err_term + var_term
    return loss


for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
    if batch>=1:
        break

    total_loss_per_molecule = 0
    n_atoms = a.size(1)
    # loop over n+1 sized subgraphs, predicting next atom & bonds
    for i in range(n_atoms): # consider 'up-to-atom-i' subgraph for all molecules at each step
        # if i>=20: break

        print(f"NODE TO PREDICT(i+1) {i+1}, CURR NODE: {i}")
        ## For every molecule, get the node labels of i+1, the 'node to predict'
        atom_labels = a[:,i+1,:]

        ## Get every molecules edge labels of node i+1 *to within current subgraph* (do not include edges to i+2,...)
        sbgr_a_i_plus_1 = a[:,:i+2,:]
        sbgr_b_i_plus_1 = b[:,:i+2,:,:]
        sbgr_e_i_plus_1 = e[:,:i+2,:]

        # nodes/connections beyond curr subgraph don't exist
        sbgr_a_i_plus_1, sbgr_b_i_plus_1, sbgr_e_i_plus_1 = remove_edges_above_node_idx(i+1, sbgr_a_i_plus_1,sbgr_b_i_plus_1,sbgr_e_i_plus_1)

        next_atom_edges = sbgr_e_i_plus_1[:,i+1,:] 
        next_atom_bonds = sbgr_b_i_plus_1[:,i+1,:,:]

        e_placeholder_col = np.full((next_atom_edges.shape[0], 1), -1)
        next_atom_edges = np.hstack((next_atom_edges, e_placeholder_col)) # add placeholder bond because '-1' reads as 'last col idx' -- this absorbs all the 'rewrite idx -1' which are otherwise impossible to get rid of; note that since it's being read as an idx all those empty values have to write to somewhere
        b_placeholder_col = np.full((next_atom_bonds.shape[0], 1, next_atom_bonds.shape[2]), -99)
        next_atom_bonds = np.concatenate([next_atom_bonds, b_placeholder_col], axis=1)

        eb_concat = np.concatenate([next_atom_edges[:, :, np.newaxis], next_atom_bonds], axis=2)
        print(f"eb concat: {eb_concat.shape}")

        num_molecules = next_atom_edges.shape[0]
        num_atoms = i+2
        num_b_feats = b.shape[3]
        
        idx_lay = np.full((eb_concat.shape[0], num_atoms, 1), -3)
        bf_lays = np.zeros((eb_concat.shape[0], num_atoms,eb_concat.shape[2]-1))
        bond_labels = np.concatenate([idx_lay, bf_lays], axis=2) # empty matrix to store rearranged bfs

        rows = np.arange(num_molecules)[:, np.newaxis]
        columns = next_atom_edges
        bond_labels[rows, columns] = eb_concat # rearrange s.t. bond feat vecs line up with idxs of destination atoms

        bond_labels = bond_labels[:, :-1,:] # remove placeholder row
        no_bond_mask = (bond_labels[:,:,0] == -3)
        no_bond_feat_layer = bond_labels[:,:,1]
        no_bond_feat_layer[no_bond_mask] = 1 # set everything that doesn't have a feature vector to 'no bond'

        # print("Edge Matrix:\n", eb_concat[:12,:,0])
        # print("Bond matrix lay 0:\n", bond_labels[:12,:,0])
        # print("Bond matrix lay 1:\n", bond_labels[:12,:,1])
        # print("Bond matrix lay 2:\n", bond_labels[:12,:,2])
        # print("Bond matrix lay 3:\n", bond_labels[:12,:,3])
        # print("Bond matrix lay 4:\n", bond_labels[:12,:,4])

        bond_labels = bond_labels[:,:,1:] # remove 'vector idx layer', leaving only bond feats

        # Final Edge Output: [mols x destination atoms x bond feats]. If there's no bond, then the bond feats will reflect that
        
        ## Dequantize labels
        bond_labels = T.from_numpy(bond_labels + np.random.rand(*bond_labels.shape))
        atom_labels = atom_labels + np.random.rand(*atom_labels.shape) # remember implicit hydrogens stored in here too

        ## Get subgraph embeddings 
        # For predicting edges: include node-to-predict but not it's bonds/bond features
        sbgr_a, sbgr_b, sbgr_e = remove_edges_above_node_idx(i, sbgr_a_i_plus_1,sbgr_b_i_plus_1,sbgr_e_i_plus_1)

        none_bond_features = T.tensor([1,0,0,0], dtype=torch.float)
        sbgr_b[:,-1,:,:] = none_bond_features

        ## Run model on i subgraph 
        # For predicting nodes:  don't include node-to-predict
        subgr_loss = 0

        mean, var = model((sbgr_a[:, :-1, :], sbgr_b[:, :-1, :, :], sbgr_e[:, :-1, :]), pred_node=True)
        # print("Output for node prediction:", mean.shape, var.shape)
        # print("# atom feats:", num_atom_features(True))

        node_loss = NLL_loss(true=atom_labels, mean=mean, var=var)
        print("Node loss:", node_loss.item())
        subgr_loss += node_loss.item()

        for target_atom in range(bond_labels.shape[1]):
            mean, var = model((sbgr_a,sbgr_b,sbgr_e), 
                                pred_node=False, 
                                idx_orig=i+1, idx_dest=target_atom)
            # print("Output for edge prediction:", mean.shape, var.shape)
            # print("# bond feats:", num_bond_features(True))

            edge_t_loss = NLL_loss(true=bond_labels[:,target_atom,:], mean=mean, var=var)
            print(f"Edge {target_atom} loss:", edge_t_loss.item())
            subgr_loss += edge_t_loss.item()
        
        print(f"Subgraph-up-to-{i} loss:", subgr_loss)
        total_loss_per_molecule += subgr_loss
    
    total_loss_per_molecule = total_loss_per_molecule / a.shape(0)  # avg loss per molecule, on every subgraph





# for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
#     print(b.shape, b)
#     if batch>=1:
#         break
#     mean, var = model((a,b,e), pred_node=True)
#     print("Output for node prediction:", mean.shape, var.shape)
#     print("# atom feats:", num_atom_features(True))

#     mean, var = model((a,b,e), pred_node=False, idx_orig=0, idx_dest=1)
#     print("Output for edge prediction:", mean.shape, var.shape)
#     print("# bond feats:", num_bond_features(True))





# fpl = fplCmd 
# hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
# layers = [num_atom_features()] + hiddenfeats
# modelParams = {
#     "fpl": fpl,
#     "activation": 'regression',
#     "conv": {
#         "layers": layers
#     },
#     "ann": {
#         "layers": layers,
#         "ba": [fplCmd, 1],
#         "dropout": df
#     }
# }
# print(f'layers: {layers}, through-shape: {list(zip(layers[:-1], layers[1:]))}')



# model = dockingProtocol(modelParams).to(device=device)
# print(model)
# # print("inital grad check")
# # for name, param in model.named_parameters():
# #     if param.requires_grad:
# #         print(name, param.data)
# totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'total trainable params: {totalParams}')
# lossFn = nn.MSELoss() # gives 'mean' val by default
# # adam, lr=0.01, weight_decay=0.001, prop=0.2, dropout=0.2
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# model.load_state_dict(torch.load('../basisModel.pth'), strict=False)
# lendl = len(trainds)
# num_batches = len(traindl)
# print("Num batches:", num_batches)
# bestVLoss = 100000000
# lastEpoch = False
# epochs = 100  # 200 initially 
# earlyStop = EarlyStopper(patience=10, min_delta=0.01)
# converged_at = 0
# trainLoss, validLoss = [], []
# trainR, validR = [], []
# for epoch in range(1, epochs + 1):
#     print(f'\nEpoch {epoch}\n------------------------------------------------')
    
#     stime = time.time()
#     model.train()
#     runningLoss, r_squared, r_list = 0, 0, []

#     for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
#         at, bo, ed, scaled_Y = a.to(device), b.to(device), e.to(device), y.to(device)

#         scaled_preds = model((at, bo, ed))
#         loss = lossFn(scaled_preds, scaled_Y)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         runningLoss += scaled_preds.shape[0] * loss.item()
 
#         preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
#         Y = scaler.inverse_transform(scaled_Y.detach().cpu().numpy().reshape(-1, 1)).squeeze()
        
#         if batch == 0:
#             print(f"Pred: {preds[:3]} vs True: {Y[:3]}")

#         _,_,r_value,_,_ = linregress(preds, Y)
#         r_list.append(r_value ** 2)

#         cStop = earlyStop.early_cstop(loss.item())
#         if cStop: break
   
#         if batch % (np.ceil(lendl / bs / 10)) == 0:
#             lossDisplay, currentDisplay = loss.item(), (batch + 1)
#             print(f'loss: {lossDisplay:>7f} [{((batch + 1) * len(a)):>5d}/{lendl:>5d}]')

#     trainLoss.append(runningLoss/lendl)
#     if len(r_list) != 0:
#         r_squared = sum(r_list)/len(r_list)
#     trainR.append(r_squared/num_batches)
#     if cStop: break
#     print(f'Time to complete epoch: {time.time() - stime}')
#     print(f'\nTraining Epoch {epoch} Results:\nloss: {runningLoss/lendl:>8f}, R^2: {r_squared/num_batches:>8f}\n------------------------------------------------')
    
#     size = len(validdl.dataset)
#     num_batches = len(validdl)
#     model.eval()
#     valid_loss = 0
#     runningLoss, r_squared, r_list = 0, 0, []
#     with torch.no_grad():
#         for (a, b, e, (scaled_y, zidValid)) in validdl:
#             scaled_preds = model((a, b, e))
#             valid_loss += lossFn(scaled_preds.to(device), scaled_y.to(device)).item()

#             preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
#             y = scaler.inverse_transform(scaled_y.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
#             _, _, r_value, _, _ = linregress(preds, y)
#             r_list.append(r_value ** 2)
#     valid_loss /= num_batches
#     if len(r_list) != 0:
#         r_squared = sum(r_list)/len(r_list)
#     validLoss.append(valid_loss)
#     validR.append(r_squared)
#     print(f'\nValidation Results:\nLoss: {valid_loss:>4f}, R^2: {r_squared:>2f}%\n------------------------------------------------')
    
#     if valid_loss < bestVLoss:
#         bestVLoss = valid_loss
#         model_path = f'r_{data}_model.pth'
#         print(f"Saved current as new best.")
#         model.save(modelParams, data, model_path, scaler)

#     if earlyStop.early_stop(valid_loss):
#         print(f'validation loss converged to ~{valid_loss}')
#         converged_at = epoch
#         break

# if cStop: 
#     print(f'training loss converged erroneously')
#     sys.exit(0)

# if converged_at != 0:
#     epochR = range(1, converged_at + 1)
# else:
#     epochR = range(1, epoch + 1)
# plt.plot(epochR, trainLoss, label='Training Loss', linestyle='-', color='lightgreen')
# plt.plot(epochR, validLoss, label='Validation Loss', linestyle='-', color='darkgreen')
# plt.plot(epochR, trainR, label='Training R^2', linestyle='--', color='lightblue')
# plt.plot(epochR, validR, label='Validation R^2', linestyle='--', color='darkblue')

# plt.title('Training and Validation Loss/R^2')
# plt.xlabel('Epochs')
# plt.ylabel('Loss / R^2')
 
# plt.legend(loc='best')

# plt.xticks(np.arange(0, epochs + 1, 2))
 
# plt.legend(loc='best')
# plt.savefig(f'./r_loss{data}.png')
# plt.show()
# plt.close()
# with open(f'./r_lossData{data}.txt', 'w+') as f:
#     f.write('train loss, validation loss\n')
#     f.write(f'{",".join([str(x) for x in trainLoss])}')
#     f.write(f'{",".join([str(x) for x in validLoss])}')