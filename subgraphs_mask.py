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
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS

from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import os
import re
import argparse


def flatten(nested, except_last=False):
    """Flattens nested tuple or list"""
    flattened_list = []

    if except_last:
      def _flatten(element):
          if isinstance(element, (list, tuple)):
            if element: # pass only on non-empty list/tup
                if all(isinstance(sub, (int, float, complex)) for sub in element):
                    flattened_list.append(element)
                else:
                    for item in element:
                        _flatten(item)
    else:
      def _flatten(element):
        if isinstance(element, (list, tuple)):
            if element:
                for item in element:
                    _flatten(item)
        else:
            flattened_list.append(element)

    _flatten(nested)
    return flattened_list


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


def draw_molecule(filename, smiles, highlight_atoms=None, color=(60.0/255.0, 80.0/255.0, 10.0/255.0)):
    figsize = (300, 300)
    highlight_color = color

    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}
    drawoptions.bgColor=None

    mol = Chem.MolFromSmiles(smiles)
    if highlight_atoms is None:
        fig = Draw.MolToMPL(mol, size=figsize, options=drawoptions,fitImage=False)
    else: 
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
        # print("ID:", ID)
        allData.set_index(ID, inplace=True)
        allData = allData.join(reference, how='left')
        allData.dropna(axis=0, inplace=True)

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
                            name=name, just_structure=False)
    return dataset


def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    # the fragment generated from smarts would have a redundant carbon, here to remove the redundant carbon
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash

def return_fg_hit_atom(mol, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash
    # sort function group atom by atom number
    sorted_all_hit_fg_at = sorted(all_hit_fg_at,
                                  key=lambda fg: len(fg),
                                  reverse=True)
    # remove small function group (wrongly matched), they are part of other big function groups
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)
    
    # wash the hit function group atom by using the remained fg, remove the small wrongly matched fg
    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import FragmentCatalog, RDConfig
import os



from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdmolops
from collections import defaultdict



# def get_murcko_scaffold_and_fragments(smiles):
#     # Convert SMILES to molecule
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError("Invalid SMILES string")

#     # Generate Murcko Scaffold
#     scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
#     # Extract rings from the scaffold
#     rings = rdmolops.GetSymmSSSR(scaffold)
    
#     # Convert scaffold to SMILES for output
#     scaffold_smiles = Chem.MolToSmiles(scaffold)
    
#     # Convert rings to SMILES
#     ring_smiles = [Chem.MolToSmiles(ring) for ring in rings]
    
#     return scaffold_smiles, ring_smiles

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

# mol = Chem.MolFromSmiles(smiles)
# ri = mol.GetRingInfo()
# atominfo = ri.AtomRings()
# print(atominfo)
# scaffold = MurckoScaffold.GetScaffoldForMol(mol)
# scaffold_atom_indices = [atom.GetIdx() for atom in scaffold.GetAtoms()]
# print("SCFF", scaffold_atom_indices)
# draw_molecule(f"test_plain.png", smiles)
# draw_molecule(f"test_rings.png", smiles, highlight_atoms=flatten(atominfo), color=(100.0/255.0, 200.0/255.0, 20.0/255.0))

# for i, ring in enumerate(atominfo):
#     ring = flatten(ring)
#     draw_molecule(f"test_ring{i}.png", smiles, highlight_atoms=ring, color=(200.0/255.0, 40.0/255.0, 20.0/255.0))

# hit_fg_at, hit_fg_name = return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)
# flattened_list = [item for sublist in hit_fg_at for item in sublist]
# hit_fg_at = [item for sublist in flattened_list for item in sublist]
# draw_molecule("test_FGs.png", smiles, highlight_atoms=hit_fg_at, color=(200.0/255.0, 40.0/255.0, 20.0/255.0))


# scaffold_smiles, ring_smiles = get_murcko_scaffold_and_fragments(smiles)
# print("Murcko Scaffold SMILES:", scaffold_smiles)
# print("Murcko Ring SMILES:", ring_smiles)




def remove_nodes(node_list,smile_i,a,b,e):
    """Remove nodes and their connections from a molecule"""
    for node_i in node_list:
        if node_i > a.shape[1]-1:
            print(f"Failed to remove atom {node_i} when from {a.shape[1]} atom molecule")
            continue
        mol_e = e[smile_i, :, :]
        mask = (mol_e == node_i)
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() == 0:
            continue

        atom_idxs = indices[:, 0]
        bond_idxs = indices[:, 1]

        # print(f"Before", e[smile_i, int(atom_idxs[0]), :]) # picks a molecule, shows u its before n after
        # print("Before edge feats:", b[smile_i, atom_idxs[0], bond_idxs[0],:])
        a[smile_i, atom_idxs, :] = 0
        b[smile_i, atom_idxs, bond_idxs, :] = 0 # -1e ensures ignored; redundancy
        e[smile_i,:,:][mask] = -1
        # print("--After:", e[smile_i, int(atom_idxs[0]), :])
        # print("==After Edge feats:", b[smile_i, atom_idxs[0], bond_idxs[0],:])


def get_ring_names(mol, ring_systems_dict):
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    found_rings = []

    if not atom_rings:
        return [], []

    for ring in atom_rings:
        ring_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=ring, kekuleSmiles=True)
        for name, pattern in ring_systems.items():
            pattern_mol = Chem.MolFromSmiles(pattern)
            if mol.HasSubstructMatch(pattern_mol):
                found_rings.append(name)
                break
    
    return atom_info, found_rings

def SME(loaded_model, orig_data, reference, scaler=None):
    dataset = setup_dataset(input_data=orig_data, name="SME", reference=reference)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                            'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                            '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                            '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                            '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                            '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                            '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                            '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                            '[C;D4]([C;D1])([C;D1])-[C;D1]',
                            '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                            '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]

    ring_systems = {
        "Benzene": "c1ccccc1",
        "Cyclohexane": "C1CCCCC1",
        "Cyclopentane": "C1CCCC1",
        "Cyclobutane": "C1CCC1",
        "Cyclopropane": "C1CC1",
        "Pyridine": "c1ccncc1",
        "Pyrrole": "c1cccn1",
        "Furan": "c1ccoc1",
        "Thiophene": "c1ccsc1",
        "Imidazole": "c1cncn1",
        "Oxazole": "c1cnco1",
        "Thiazole": "c1cscn1",
        "Indole": "c1ccc2c(c1)ccn2",
        "Isoquinoline": "c1ccc2c(c1)ccnc2",
        "Quinoline": "c1ccc2nc3ccccc3cc2c1",
        "Naphthalene": "c1ccc2ccccc2c1",
        "Anthracene": "c1ccc2cc3ccccc3cc2c1",
        "Phenanthrene": "c1ccc2c(c1)ccc3ccccc23"
        # temp dataset
    }

    molecule_data = defaultdict(lambda: defaultdict(float))
    subgraph_improvements = defaultdict(lambda: {'sum': 0.0, 'atoms': []})

    loaded_model.eval()
    for batch, (a, b, e, (y, zID)) in enumerate(dataloader):
        if batch>20: break
        if batch<100:
            # smile = dataset.smiles[batch]
            print("ID (should be smile):", zID)
        # at, bo, ed, Y = a.to(device), b.to(device), e.to(device), y.to(device)

        # get functional groups
        mol = Chem.MolFromSmiles("ClCCC#N")
        hit_fg_at, hit_fg_name = return_fg_hit_atom(mol, fg_name_list, fg_with_ca_list, fg_without_ca_list)
        print("Hit atoms:", hit_fg_at, "\n Hit names:", hit_fg_name)
        draw_molecule("test_FGs.png", "ClCCC#N", highlight_atoms=flatten(hit_fg_at), color=(200.0/255.0, 40.0/255.0, 20.0/255.0))

        # get rings
        ring_ats, ring_names = get_ring_names(mol, ring_systems)

        print("fgat", hit_fg_at, "ringat", ring_ats)
        subgr_at = flatten(hit_fg_at, except_last=True) + flatten(ring_ats, except_last=True)
        print("subgrat", subgr_at)

        # [[1],[3],[9]] -- diff instances of '-F' group
        # ## todo: add in combos

        print("Num subgraphs:", len(subgr_at))
        masked_a = a.repeat(1+len(subgr_at),1,1)  # copy the smile n times
        masked_b = b.repeat(1+len(subgr_at),1,1,1)
        masked_e = e.repeat(1+len(subgr_at),1,1)

        for copy_i, atom_group in enumerate(subgr_at, start=1):
            remove_nodes(atom_group, copy_i, masked_a, masked_b, masked_e)

        with torch.no_grad():
            preds = loaded_model((masked_a, masked_b, masked_e)).reshape(-1, 1)
            print("Preds:", preds)
            print("A - shape:", masked_a.shape)
            if scaler:
                preds = scaler.inverse_transform(preds)
                print("Scaled pred:", preds)

        # old - new = difference
        original = preds[0]
        preds[1:] = preds[1:] - original

        # scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # scaffold_atom_indices = [atom.GetIdx() for atom in scaffold.GetAtoms()]
        # print("SCFF", scaffold_atom_indices)
        # draw_molecule(f"test_plain.png", smiles)
        # draw_molecule(f"test_rings.png", smiles, highlight_atoms=flatten(atominfo), color=(100.0/255.0, 200.0/255.0, 20.0/255.0))

input = "dock_acease_pruned.txt"

# reference SMILEs/zID
smileData = pd.read_csv('./data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smile', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

# import model
model_path = "/data/users/vantilme1803/nfp-docking/src/trainingJobs/r_dock_glprors_model.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = dockingProtocol(params=checkpoint['params'])
model.load_state_dict(checkpoint['model_state_dict'])

SME(model, input, smileData, checkpoint['scaler'])