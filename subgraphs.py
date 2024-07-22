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
from networkP import dockingProtocol, GraphLookup
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

def contains_problematic_chars(filename):
    # Define a regex pattern for problematic characters (e.g., slashes, backslashes)
    pattern = r'[\/\\]'
    return bool(re.search(pattern, filename))

def group_most_associated_w_fp_feature(fp_i, fp_size, degree_activations, model_data_scaler):
    """Produce a dict of {Molecule: group (atom-radii) that most activates a specific fingerprint feature}'s"""
    most_associated_activations = {}

    # for each molecule, find the degree-atom pair with the highest value at the fingerprint feature index
    for degree, activations_dict in degree_activations.items():
        for ID, tensor in activations_dict.items():
            assert fp_size == tensor.shape[1], f"fp_size {fp_size} does not match hidden layer feature size {tensor.shape[1]}"
            assert fp_i < tensor.shape[1], f"Index {fp_i} is out of bounds for tensor shape {tensor.shape[1]}"
            
            assoc_values = tensor[:,fp_i]
            max_index = torch.argmax(assoc_values).item()
            max_activations = tensor[max_index, :]

            if ID not in most_associated_activations:
                most_associated_activations[ID] = (degree, max_activations, max_index)
            else:
                old_max = most_associated_activations[ID][1][fp_i]
                new_max = max_activations[fp_i]
                if new_max>old_max:
                    most_associated_activations[ID] = (degree, max_activations, max_index)

    return most_associated_activations


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
    a,_,_ = buildFeats(smile)
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


def setup_dataset(input_data, name, reference, input_only=False, no_graph=False):
    # input zIDs
    data_path = find_item_with_keywords(search_dir='./data',keywords=[input_data],file=True)[0]
    # print("Data path:", data_path)
    allData = labelsToDF(data_path)
    # print("Data cols:", allData.columns)

    # Ensure consistent 'smile' label; or get smiles from ZID reference file
    smile_names = ['smiles','smile','SMILEs','SMILES','SMILE']
    if any(option in allData.columns for option in smile_names):
        for option in smile_names:
            if option in allData.columns:
                allData.rename(columns={option: 'smile'}, inplace=True)

    ID = get_ID_type(allData)
    if ID == 'smile':
        allData.set_index(ID, inplace=True, drop=False)
    else:
        print("ID:", ID)
        allData.set_index(ID, inplace=True)
        print("ALLDATA:",allData)
        print("reference :", reference)
        allData = allData.join(reference, how='left')
        print("ALLDATA:",allData)


    xData = [[index, row['smile']] for index, row in allData.iterrows()] # (ID, smile)
    if input_only:
        yData = [0] * len(xData)
    else: 
        yData = allData['labels'].values.tolist()

    if no_graph:
        if input_only: return xData
        return xData, yData

    dataset = dockingDataset(train=xData, 
                            labels=yData,
                            name=name)
    return dataset



def find_most_predictive_features(loaded_model, orig_data, reference):
    """Compare each fp feature vs labels, get R^2, return most anti/correlated features"""

    # calculate fingerprints for original dataset
    dataset = setup_dataset(input_data=orig_data, name="Calc. FP's", reference=smileData)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    fp_dict = {}
    for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
            at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)
            _, fps = loaded_model((at, bo, ed), return_fp=True)
            fps = fps.detach().numpy()

            for i, z_id in enumerate(zID):
                fp_dict[z_id] = fps[i]

    # calculate (labels, feature) correlations
    orig_data_paths = find_item_with_keywords(search_dir="./data", keywords=[orig_data], file=True)
    orig_data_path = min(orig_data_paths, key=len)
    origData = labelsToDF(orig_data_path)
    ID = get_ID_type(origData)
    origData.set_index(ID, inplace=True)
    np_index = origData.index.values
    np_array = origData['labels'].values.reshape(-1, 1)

    m = np_index.shape[0]
    first_fp = next(iter(fp_dict.values()))
    fp_len = first_fp.shape[0]
    n = fp_len
    fp_arr = np.zeros((m, n)) # num-molecules x num-fingerprint-features

    for i, ID in enumerate(np_index):
        if ID not in fp_dict:
            print(f"Fingerprint not found for molecule {i} in original dataset.")
            continue
        fp = fp_dict[ID] # makes sure ID<->(ID,fp) aligned
        fp_arr[i,:] = fp
    merged_arr = np.concatenate([np_array, fp_arr], axis=1)

    corr_list = []
    for i, fp_feature in enumerate(merged_arr[:, 1:].T):  
        labels = merged_arr[:,0]
        slope, intercept, r_value, p_value, std_err = linregress(labels, fp_feature)
        r_squared = r_value ** 2
        corr_list.append({'Feature #': i, 'R': r_value, 'R^2': r_squared, 'P': p_value})

    max_R_feat = max(corr_list, key=lambda x: x['R'])
    max_Rsquared_feat = max(corr_list, key=lambda x: x['R^2'])
    min_R_feat = min(corr_list, key=lambda x: x['R'])

    print(f"---- For {orig_data} model ----\nMost corr:{max_R_feat}\nMost anticorr:{min_R_feat}")
    return max_R_feat, min_R_feat

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

input = "sol_data_ESOL"

# reference SMILEs/zID
smileData = pd.read_csv('./data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smile', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

dataset = setup_dataset(input_data=target_dataset, name="Get conv. activations", reference=smileData, input_only=True)
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

# import model
model_path = find_item_with_keywords(search_dir='./src/trainingJobs',keywords=[apply_model],file=True)[0]
print("Applying model: ", model_path, "\n - on dataset: ", find_item_with_keywords(search_dir='./data',keywords=[target_dataset],file=True)[0])
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = dockingProtocol(params=checkpoint['params'])
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']

fp_dict = {}
degree_activations = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
# feed SMILEs into model, get out activations
for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
        at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)
        activs, fps, preds = model((at, bo, ed), return_conv_activs=True, return_fp=True)
        fps = fps.detach().numpy()

        for i, z_id in enumerate(zID):
            fp_dict[z_id] = fps[i]

        for degreeTuple in activs:
            degree = degreeTuple[0]
            vec = degreeTuple[1]
            assert degree in degree_activations, f"Unexpected degree: {degree}"

            activation_dict = degree_activations[degree]
            for i, z_id in enumerate(zID):
                activation_dict[z_id] = vec[i,:,:]

best_feat, worst_feat = find_most_predictive_features(model, checkpoint['dataset'], smileData)
first_fp = next(iter(fp_dict.values()))
fp_len = first_fp.shape[0]

best_dict = group_most_associated_w_fp_feature(best_feat['Feature #'],fp_len,degree_activations,scaler)
worst_dict = group_most_associated_w_fp_feature(worst_feat['Feature #'],fp_len,degree_activations,scaler)

output_dir = os.path.join(os.getcwd(), 'results', checkpoint['dataset'])
os.makedirs(output_dir, exist_ok=True)
best_path, worst_path = os.path.join(output_dir, 'best'), os.path.join(output_dir, 'worst')
for subdir in [best_path, worst_path]:
    os.makedirs(subdir, exist_ok=True)

for i, (ID, atomTuple) in enumerate(best_dict.items()):
    if 'ZINC' in ID: # get smiles from reference
        print(f"ZID:{ID}")
        smile = get_smile_from_zinc_id(ID, smileData)
    else:
        if i==0: # get smiles from orig dataset
            orig_IDs = setup_dataset(input_data=checkpoint['dataset'], name="Retrieve SMILEs", reference=smileData, input_only=True, no_graph=True)
            orig_IDs = np.array(orig_IDs)
            orig_IDs = pd.DataFrame(orig_IDs, index=orig_IDs[:, 0], columns=['ID', 'smile'])
        try:
            smile = orig_IDs.loc[ID][1]
        except KeyError:
            smile = None
            print(f"Smile not found for{ID}")
            continue
    
    degree = atomTuple[0]
    atom_index = atomTuple[2]
    atom_neighborhood = get_atom_neighborhood([smile], atom_index, degree)

    if i>=50: break
    
    print(f"Molecule {i}:", smile, "- best atoms:", atom_neighborhood)
    ID_name = ''.join(ID.split())

    color = (40.0/255.0, 200.0/255.0, 80.0/255.0)

    if not contains_problematic_chars(smile):
        save_to = os.path.join(best_path, f"best_{smile}.png")
        draw_molecule_with_highlights(save_to, smile, atom_neighborhood,color) # note both are RDKit ordering, indices align


for i, (ID, atomTuple) in enumerate(worst_dict.items()):
    if 'ZINC' in ID: # get smiles from reference
        print(f"ZID:{ID}")
        smile = get_smile_from_zinc_id(ID, smileData)
    else:
        if i==0: # get smiles from orig dataset
            orig_IDs = setup_dataset(input_data=checkpoint['dataset'], name="Retrieve SMILEs", reference=smileData, input_only=True, no_graph=True)
            orig_IDs = np.array(orig_IDs)
            orig_IDs = pd.DataFrame(orig_IDs, index=orig_IDs[:, 0], columns=['ID', 'smile'])
        try:
            smile = orig_IDs.loc[ID][1]
        except KeyError:
            smile = None
            print(f"Smile not found for{ID}")
            continue
    
    degree = atomTuple[0]
    atom_index = atomTuple[2]
    atom_neighborhood = get_atom_neighborhood([smile], atom_index, degree)

    if i>=50: break

    print(f"Molecule {i}:", smile, "- worst atoms:", atom_neighborhood)
    ID_name = ''.join(ID.split())

    color = (200.0/255.0, 40.0/255.0, 20.0/255.0)

    if not contains_problematic_chars(smile):
        save_to = os.path.join(worst_path, f"worst_{smile}.png")
        draw_molecule_with_highlights(save_to, smile, atom_neighborhood,color) # note both are RDKit ordering, indices align
