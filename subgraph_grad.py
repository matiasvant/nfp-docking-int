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
from networkP import dockingProtocol, GraphLookup, EnsembleReg, nfpConv
from util import *
import time
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import os
import re
import argparse
import pickle
import warnings

def contains_problematic_chars(filename):
    # Define a regex pattern for problematic characters (e.g., slashes, backslashes)
    pattern = r'[\/\\]'
    return bool(re.search(pattern, filename))

def get_smile_from_zinc_id(zinc_id, reference):
    ## note to self: when construct new synthetic mols, will need to handle z_id/smile labeling; maybe add to master-data-dock & give me names "SYNTHETIC1020"    
    try:
        smile = smileData.loc[zinc_id, 'smile']
        return smile
    except KeyError:
        print(f"ZINC ID {zinc_id} not found.")
        return None


def get_smile_from_dataset(ID, DataFrame):
    try:
        smile = smileData.loc[ID, 'smile']
        return smile
    except KeyError:
        print(f"ID {ID} not found.")
        return None


def get_atom_neighborhood(smile, center_atom_i, max_degree):
    # max deg = 0 (just central), = 1 (central+first neighbors), ...
    _,_,e = buildFeats(smile)
    atom_neighborhood = [center_atom_i]

    for neighbor_degree in range(max_degree):
        for atom in list(atom_neighborhood): # iter over a copy to avoid reading neighbors as they're added
            neighbors = e[0, atom, :]
            for neighbor_slot in neighbors:
                neighbor_i = neighbor_slot.item()
                if neighbor_i != -1 and neighbor_i not in atom_neighborhood:  # -1 == neighbor doesn't exist
                    atom_neighborhood.append(neighbor_i)

    return atom_neighborhood


def draw_molecule_with_highlights(filename, smiles, highlight_atoms, color=(60.0/255.0, 80.0/255.0, 10.0/255.0) ):
    figsize = (300, 300)
    highlight_color = color

    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}
    drawoptions.bgColor=None

    mol = Chem.MolFromSmiles(smiles)
    fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, size=figsize, options=drawoptions,fitImage=False)

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def setup_dataset(input_data, name, reference, input_only=False, no_graph=False, strip_ID=False):
    # input zIDs
    data_path = find_item_with_keywords(search_dir='./data',keywords=[input_data],file=True)[0]
    allData = labelsToDF(data_path) # labels, compound id, smiles
    # print("Data cols:", allData.columns)

    # Ensure consistent 'smile' label; or get smiles from ZID reference file
    smile_names = ['smiles','smile','SMILEs','SMILES','SMILE']
    if any(option in allData.columns for option in smile_names):
        for option in smile_names:
            if option in allData.columns:
                allData.rename(columns={option: 'smile'}, inplace=True)

    ID = get_ID_type(allData)
    if ID == 'smile':
        allData.set_index(ID, inplace=True)
    else:
        allData.set_index(ID, inplace=True)
        if 'smile' not in allData.columns:
            allData = allData.join(reference, how='left')
        allData.dropna(axis=0, inplace=True)
        if strip_ID:
            allData.set_index('smile', inplace=True)

    just_smiles = False
    if ID == 'smile' or strip_ID:
        just_smiles = True
        xData = [index for index, row in allData.iterrows()]
    else:
        xData = [[index, row['smile']] for index, row in allData.iterrows()] # (ID, smile)
    
    if input_only:
        yData = [0] * len(xData)
    else: 
        yData = allData['labels'].values
        yData = yData.reshape(-1, 1)
        scaler = StandardScaler()
        yData = scaler.fit_transform(yData).T[0].tolist() # since reg_train/ensemble splits are random, approx. them by scaling over whole dataset  

    if no_graph:
        if input_only: return xData
        return xData, yData
    
    allData.reset_index(inplace=True)
    xData = [[index, row['smile']] for index, row in allData.iterrows()] # (ID, smile)

    # (ID, smile), label
    dataset = dockingDataset(train=xData, 
                            labels=yData,
                            name=name, just_structure=False, atom_masks=None, just_smiles=just_smiles)
    return dataset, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--m', type=str, required=True)
    parser.add_argument('-eval_dataset', '--d', type=str, required=True)
    args = parser.parse_args()
    apply_model, target_dataset = args.m, args.d


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # reference SMILEs/zID
    smileData = pd.read_csv('./data/smilesDS.smi', delimiter=' ')
    smileData.columns = ['smile', 'zinc_id']
    smileData.set_index('zinc_id', inplace=True)

    dataset, scaler = setup_dataset(input_data=target_dataset, name="Get gradients", reference=smileData, input_only=False)
    dataloader = DataLoader(dataset, batch_size=7, shuffle=False)

    # import ensemble model
    model_path = find_item_with_keywords('./src/ensemble', ['model'], dir=False, file=True)
    model_path = model_path[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(model_path)
    ensemble = EnsembleReg(checkpoint['params']['num_models'], *checkpoint['params']['models']).to(device)
    ensemble.load_state_dict(checkpoint['model_state_dict'])
    lossFn = nn.MSELoss()
    for name,param in ensemble.named_parameters(): # remove pre-classifier-freeze applied at train
        param.requires_grad = True

    all_grads = {}
    one_mols_grad = []
    hooks = []

    # For a submodel: get the output of lastconv via forward hook; get its grads via backward hook
    def wrapper(module, model_i):
        def submodel_forward_hook(module, input, output):     
            def output_backward_hook(grad):
                # print(f"Model {model_i} grad-shape:", grad.shape)
                one_mols_grad.append(grad.squeeze(0))
            back_hook = output.register_hook(output_backward_hook)
            hooks.append(back_hook)
            
        forward_hook = module.register_forward_hook(submodel_forward_hook)
        hooks.append(forward_hook)

    for name, mod in ensemble.named_modules():
        # attach hook to last conv layer of each submodel
        if isinstance(mod, nfpConv) and 'layersArr.3' in name:
            model_i = ".".join(name.split(".", 2)[:2])
            wrapper(mod, model_i)

    ensemble.eval()
    for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
            at, bo, ed, scaled_Y = a.to(device), b.to(device), e.to(device), y.to(device)

            for i in range(y.shape[0]):
                one_mols_grad = []
                scaled_pred = ensemble((at[i,:,:].unsqueeze(0), bo[i,:,:].unsqueeze(0), ed[i,:,:].unsqueeze(0)))
                loss = lossFn(scaled_pred.squeeze(), scaled_Y[i])
                loss.backward(retain_graph=True)
                all_grads[zID[i]] = one_mols_grad

            # preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
            # Y = scaler.inverse_transform(scaled_Y.detach().cpu().numpy().reshape(-1, 1)).squeeze()
            if batch>2: break

    # print("IDs, gradnorm per [1,2,3,4] models")
    for mol,grads in all_grads.items():
        print(mol)
        for grad in grads:
            print("Grad:", grad.shape, torch.norm(grad))

    for hook in hooks:
        hook.remove()

    