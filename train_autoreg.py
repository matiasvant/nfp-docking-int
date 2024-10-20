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

def remove_edges_above_node_idx(idx,a,b,e):
    # find connections to atoms beyond current specified node/subgraph
    threshold_val = idx
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
    b[mol_idxs, atom_idxs, bond_idxs,:] = none_bond_features # -1 ensures ignored; redundancy
    e[mask] = -1
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


def NLL_loss(true, mean, var, visualize=False):
    ## Error term: Sample gaussian density at (true,μ,σ2) to find how unlikely, then flip via log
    # simplified -log(Gaussian PDF), rid of constants (irrelevant for optimization) = 1/2(log(σ2) + ((true-μ)^2 / σ2))
    incentivize_var = 1
    var = torch.maximum(var, torch.tensor(1e-1, dtype=var.dtype, device=var.device))  # numerical stability
    logvar = torch.log(var) * incentivize_var
    err = ((true - mean) ** 2) / (var*incentivize_var)
    err_loss = 0.5 * (logvar + err)
    err_loss = err_loss.mean()

    # ## Var term
    # one_over_var = 1.0 / var
    # # print("1/var:", one_over_var)
    # var_loss = -torch.log(one_over_var)
    # # print("var_loss:", var_loss)
    # var_loss = var_loss.mean()
    # # print("var_loss mean:", var_loss)

    if visualize:
        print("Var:", var[0,:5].detach())
        print("log(var):", logvar.mean().item())
        print("true-mean/var:", err.mean().item())
        # print("Final Var loss:", var_loss)

    return err_loss

def check_accuracy(label, pred):
    """Check quantized feature prediction accuracy (non-sampled) for dequantized label"""
    true_one_hot = np.argmax(label.detach().numpy(), axis=1)
    pred_one_hot = np.argmax(pred.detach().numpy(), axis=1)
    matching_rows = (true_one_hot == pred_one_hot).sum().item()
    total_rows = len(true_one_hot)
    matching_ratio = matching_rows / total_rows
    return matching_ratio

def b_downsample(data):
    new_list = []
    moving_sum = 0
    len_between = 0
    for i, elem in enumerate(data):
      if isinstance(elem, (float,int)):
        moving_sum += elem
        len_between += 1
      if elem == 'B':
        if len_between != 0:
            new_list.append(moving_sum/len_between)
      if elem == 'E':
        new_list.append('E')
    return new_list


fpl = fplCmd 
hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
layers = [num_atom_features(just_structure=True)] + hiddenfeats
params = {
    "1var": False,
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
print("Model structure:\n", model)

# # print("inital grad check")
# # for name, param in model.named_parameters():
# #     if param.requires_grad:
# #         print(name, param.data)
# totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'total trainable params: {totalParams}')

# # adam, lr=0.01, weight_decay=0.001, prop=0.2, dropout=0.2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
lendl = len(trainds)
num_batches = len(traindl)
print("Num batches:", num_batches)
bestVLoss = 100000000
lastEpoch = False
epochs = 10  # 50
earlyStop = EarlyStopper(patience=10, min_delta=0.01)
converged_at = 0
trainLoss, validLoss, rsq_list = [], [], []

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

loss = 'CE'
edge_pred = True
node_pred = True

all_loss, all_n_r, all_e_r, all_n_match, all_e_match = [], [], [], [], []

for epoch in range(1, epochs + 1):
    print(f'\nEpoch {epoch}\n------------------------------------------------')
    
    stime = time.time()
    model.train()
    epoch_loss = 0

    all_loss.append('E')
    all_n_r.append('E')
    all_n_match.append('E')
    all_e_r.append('E')
    all_e_match.append('E')

    # if epoch>1:
    #     for name, param in model.Node_Pred.mean_mlp.mlp.named_parameters():
    #         print(f'At-Start-of-Epoch {epoch}- MLP {name} requires_grad: {param.requires_grad}')

    for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
        # if batch>200: break
        a,b,e = a.to(device), b.to(device), e.to(device)

        max_atoms = a.size(1)
        r_list = []
        batch_loss = torch.tensor([0.0], requires_grad=True)

        for i in range(1, max_atoms-1): # consider 'up-to-atom-i' subgraph for all molecules at each step
            ## For every molecule/graph, get the node labels of i+1, the 'node to predict'
            atom_labels = a[:,i+1,:]

            ## Get every molecules edge labels of node i+1 *to within current subgraph* (do not include edges to i+2,...)
            sbgr_a_i_plus_1 = a[:,:i+2,:]
            sbgr_b_i_plus_1 = b[:,:i+2,:,:]
            sbgr_e_i_plus_1 = e[:,:i+2,:]

            # nodes/connections beyond curr subgraph don't exist
            sbgr_a_i_plus_1, sbgr_b_i_plus_1, sbgr_e_i_plus_1 = remove_edges_above_node_idx(i+1, sbgr_a_i_plus_1,sbgr_b_i_plus_1,sbgr_e_i_plus_1)

            # only predict for molecules which have a next-atom/didn't end
            valid_atoms_mask = torch.sum(a[:, i+1, :], dim=1) != 0
            if valid_atoms_mask.sum() == 0:
                continue

            atom_labels = atom_labels[valid_atoms_mask]
            sbgr_a_i_plus_1 = sbgr_a_i_plus_1[valid_atoms_mask]
            sbgr_b_i_plus_1 = sbgr_b_i_plus_1[valid_atoms_mask]
            sbgr_e_i_plus_1 = sbgr_e_i_plus_1[valid_atoms_mask]

            next_atom_edges = sbgr_e_i_plus_1[:,i+1,:]
            next_atom_bonds = sbgr_b_i_plus_1[:,i+1,:,:]

            e_placeholder_col = np.full((next_atom_edges.shape[0], 1), -1)
            next_atom_edges = np.hstack((next_atom_edges, e_placeholder_col)) # add placeholder bond because '-1' reads as 'last col idx' -- this absorbs all the 'rewrite idx -1' which are otherwise impossible to get rid of; note that since it's being read as an idx all those empty values have to write to somewhere
            b_placeholder_col = np.full((next_atom_bonds.shape[0], 1, next_atom_bonds.shape[2]), -99).astype(np.float32)
            next_atom_bonds = np.concatenate([next_atom_bonds, b_placeholder_col], axis=1)

            eb_concat = np.concatenate([next_atom_edges[:, :, np.newaxis], next_atom_bonds], axis=2)

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
            bond_labels = torch.from_numpy(bond_labels)

            # Final Edge Output: [mols x destination atoms x one-hot bond feats]. No-bond is an option
            
            ## Dequantize labels
            nd_a = atom_labels

            # bond_labels = T.from_numpy(bond_labels + np.random.rand(*bond_labels.shape)).float()
            # atom_labels = atom_labels + np.random.rand(*atom_labels.shape).astype(np.float32) # remember implicit hydrogens stored in here too

            ## Get subgraph embeddings 
            # For predicting edges: include node-to-predict but not it's bonds/bond features
            sbgr_a, sbgr_b, sbgr_e = remove_edges_above_node_idx(i, sbgr_a_i_plus_1,sbgr_b_i_plus_1,sbgr_e_i_plus_1)

            none_bond_features = T.tensor([1,0,0,0], dtype=torch.float)
            sbgr_b[:,-1,:,:] = none_bond_features

            ## Run model on i subgraph 
            # For predicting nodes:  don't include node-to-predict

            if i==1:
                all_loss.append('B')
                all_n_r.append('B')
                all_e_r.append('B')
                all_n_match.append('B')
                all_e_match.append('B')

            node_loss = torch.tensor([0], device=device).float()
            if node_pred:
                mean = model((sbgr_a[:, :-1, :], sbgr_b[:, :-1, :, :], sbgr_e[:, :-1, :]), pred_node=True)
                # mean = torch.zeros_like(atom_labels)
                # mean[:,0] = 1  # compare to 'always guess carbon'

                if loss == 'NLL':
                    node_loss = NLL_loss(true=atom_labels, mean=mean, var=var)
                if loss == 'MSE':
                    node_loss = mse_loss(atom_labels[:,:43], mean[:,:43])
                if loss == 'CE':
                    __, ind_labels = atom_labels[:,:43].max(dim=1) # reduce to one-hot label
                    node_loss = ce_loss(mean[:,:43], ind_labels)

                # _, _, r_value, _, _= linregress(atom_labels.detach().numpy().flatten(), mean.detach().numpy().flatten())
                r_value=1

                # Check prediction accuracy (non-sampled)
                matching_ratio = check_accuracy(atom_labels[:,:43], mean[:,:43]) # compare one-hot atom choices
                
                
                r_list.append(r_value ** 2)
                all_n_r.append(r_value ** 2)
                all_n_match.append(matching_ratio)

                if i==1:
                    if loss == 'NLL':
                        node_loss = NLL_loss(true=atom_labels, mean=mean, var=var, visualize=True)
                        print("Var:", var[0,:5])
                    print(f"Batch {batch} - Subgraph-up-to-{i}:")
                    print(f"True: {atom_labels[0,:10]}, \n Pred: {mean[0,:10]}")

            edge_loss = torch.tensor([0], device=device).float()
            e_r, e_ratio = [], []
            if edge_pred:
                for target_atom in range(bond_labels.shape[1]):
                    if target_atom > 5: break

                    # edge-pred easier, train on a small portion of batch to save compute
                    batch_subset = 5
                    
                    pred = model((sbgr_a[:batch_subset,:,:],sbgr_b[:batch_subset,:,:,:],sbgr_e[:batch_subset,:,:]), 
                                        pred_node=False, 
                                        idx_orig=i+1, idx_dest=target_atom)
                    dest_target_bond_features = bond_labels[:batch_subset,target_atom,:]
                    if loss == 'NLL':
                        edge_t_loss = NLL_loss(true=dest_target_bond_features.float(), mean=pred.float(), var=var.float())
                    if loss == 'MSE':
                        edge_t_loss = mse_loss(dest_target_bond_features.float(), pred.float())
                    if loss == 'CE':
                        __, ind_labels = dest_target_bond_features.max(dim=1)
                        edge_t_loss = ce_loss(pred, ind_labels)

                        if (batch%20) and (i==1 or i==20):
                            print(f"-- Target Atom {target_atom}, Batch {batch} Atom {i} --")
                            # print("Bond feats (64 x 4):", dest_target_bond_features[:5,:])
                            print("Labels:", ind_labels[:5])
                            print("Preds:", pred[:5].max(dim=1).indices)
    

                    edge_loss = edge_loss + edge_t_loss
                    # _, _, r_value, _, _= linregress(dest_target_bond_features.detach().numpy().flatten(), mean.detach().numpy().flatten())
                    r_value = 1
                    matching_ratio = check_accuracy(dest_target_bond_features[:batch_subset,:], pred)
                    e_r.append(r_value ** 2)
                    e_ratio.append(matching_ratio)

                edge_loss = edge_loss / bond_labels.shape[1] # weight node/edge preds roughly equally
                all_e_match.append(sum(e_ratio)/len(e_ratio))
                all_e_r.append(sum(e_r)/len(e_r))

            subgr_loss = node_loss + edge_loss
            all_loss.append(subgr_loss.item())
            subgr_loss = (subgr_loss*(atom_labels.shape[0]/a.shape[0])) # CE,MSE,NLL implementation weights all equally; as batches get smaller, weight loss lower

            batch_loss = batch_loss + subgr_loss

        batch_loss = 10 * batch_loss/a.shape[0] # normalize by batchsize
        epoch_loss += batch_loss.item()


        optimizer.zero_grad()
        print(f"Batch {batch} loss: {batch_loss.item()}")
        batch_loss.backward()

        optimizer.step()


    print(f"Epoch {epoch} Loss - {epoch_loss}")
    trainLoss.append(epoch_loss)
    if node_pred: 
        avg_rsq = sum(r_list)/len(r_list)
        print(f"Avg R^2: {avg_rsq}. Per subgraph (should be increasing): {r_list}")
        rsq_list.append(avg_rsq)

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

if converged_at != 0:
    epochR = range(1, converged_at + 1)
else:
    epochR = range(1, epoch + 1)

if params["1var"]:
    var_t = "1_Var"
else:
    var_t = "Var"
if node_pred and edge_pred: pred_task = "n&e"
elif node_pred: pred_task = "node"
elif edge_pred: pred_task = "edge"

model_path = f'AR_{data}_model_{var_t}_{loss}_{pred_task}.pth'
print(f"Saved model at epoch {epoch}.")
model.save(params, model_path)

## Loss plot
print("all loss:", all_loss)
all_loss = b_downsample(all_loss)
all_loss_numeric = [x if not isinstance(x,str) else np.nan for x in all_loss]
preds = np.arange(0, len(all_loss))

print("Preds:", preds.dtype, preds)
print("All Loss:", all_loss_numeric)
plt.plot(preds, all_loss_numeric, label=f'{pred_task} Prediction Loss', linestyle='-', color='lightgreen')

# mark batch/epoch transitions
for i, value in enumerate(all_loss):
    if value == 'B':
        plt.axvline(x=preds[i], color='gray', linestyle='--', linewidth=0.5)
    if value == 'E':
        plt.axvline(x=preds[i], color='purple', linestyle='--', linewidth=0.5)

plt.title(f'{loss} Full Training Loss')
plt.xlabel('Predictions')
plt.ylabel(f'Loss ({loss})')

plt.legend(loc='best')
num_ticks = 8
step = max(1, int(round(len(preds) / num_ticks)))
plt.xticks(np.arange(0, len(preds), step))


plt.savefig(f'autoreg_loss_{loss}_{var_t}_{pred_task}.png')
plt.show()
plt.close()


## R plot
all_n_r = b_downsample(all_n_r)
all_e_r = b_downsample(all_e_r)
all_n_r_numeric = [x if not isinstance(x,str) else np.nan for x in all_n_r]
all_e_r_numeric = [x if not isinstance(x,str) else np.nan for x in all_e_r]
npreds, epreds = np.arange(0, len(all_n_r_numeric)), np.arange(0, len(all_e_r_numeric))

if node_pred: plt.plot(npreds, all_n_r_numeric, label='Node Prediction R^2', linestyle='-', color='blue')
if edge_pred: plt.plot(epreds, all_e_r_numeric, label='Edge Prediction R^2', linestyle='-', color='green')

for i, value in enumerate(all_loss):
    if value == 'B':
        plt.axvline(x=preds[i], color='gray', linestyle='--', linewidth=0.5)
    if value == 'E':
        plt.axvline(x=preds[i], color='purple', linestyle='--', linewidth=0.5)

plt.title(f'{loss} Full Training R^2')
plt.xlabel('Predictions')
plt.ylabel('R^2')

plt.legend(loc='best')
plt.xticks()
# plt.xticks(np.arange(0, len(preds), int(round(len(preds)/num_ticks))))
plt.yticks(np.linspace(0, 1, 8))

# all_n_r_numeric = np.array(all_n_r_numeric)
# all_n_r_numeric = all_n_r_numeric[~np.isnan(all_n_r_numeric)]
# y_min = min(min(all_n_r_numeric), min(all_e_r_numeric))
# y_max = max(max(all_n_r_numeric), max(all_e_r_numeric))
# plt.yticks(np.linspace(y_min, y_max, num_ticks))

plt.savefig(f'autoreg_R^2_{loss}_{var_t}_{pred_task}.png')
plt.show()
plt.close()


## Match plot
all_n_match = b_downsample(all_n_match)
all_e_match = b_downsample(all_e_match)
all_n_match_numeric = [x if not isinstance(x,str) else np.nan for x in all_n_match]
all_e_match_numeric = [x if not isinstance(x,str) else np.nan for x in all_e_match]
nm_preds, em_preds = np.arange(0, len(all_n_match_numeric)), np.arange(0, len(all_e_match_numeric))

if node_pred: plt.plot(nm_preds, all_n_match_numeric, label='Node Prediction Accuracy %', linestyle='-', color='blue')
if edge_pred: plt.plot(em_preds, all_e_match_numeric, label='Edge Prediction Accuracy %', linestyle='-', color='green')

for i, value in enumerate(all_loss):
    if value == 'B':
        plt.axvline(x=preds[i], color='gray', linestyle='--', linewidth=0.5)
    if value == 'E':
        plt.axvline(x=preds[i], color='purple', linestyle='--', linewidth=0.5)

plt.title(f'{loss} Full Accuracy %')
plt.xlabel('Predictions')
plt.ylabel('% Accuracy')

plt.legend(loc='best')
num_ticks = 8
plt.xticks()
# plt.xticks(np.arange(0, len(preds), int(round(len(preds)/num_ticks))))

plt.savefig(f'autoreg_match_{loss}_{var_t}_{pred_task}.png')
plt.show()
plt.close()

