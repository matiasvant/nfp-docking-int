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
from itertools import combinations
import pickle
from tqdm import tqdm
from subgraphs import setup_dataset, get_atom_neighborhood

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


def draw_molecule(filename, smiles, highlight_atoms=None, color=(60.0/255.0, 80.0/255.0, 10.0/255.0), mol=None):
    figsize = (300, 300)
    highlight_color = color

    drawoptions = DrawingOptions()
    drawoptions.selectColor = highlight_color
    drawoptions.elemDict = {}
    drawoptions.bgColor=None

    if not mol:
        mol = Chem.MolFromSmiles(smiles)
    if highlight_atoms is None:
        fig = Draw.MolToMPL(mol, size=figsize, options=drawoptions,fitImage=False)
    else: 
        fig = Draw.MolToMPL(mol, highlightAtoms=highlight_atoms, size=figsize, options=drawoptions,fitImage=False)

    fig.gca().set_axis_off()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def all_combinations(lst, lists=False):
    result = []

    if lists:
        for r in range(1, len(lst) + 1):
            for comb in combinations(lst, r):
                result.append([item for sublist in comb for item in sublist])
        return result

    else:
        for r in range(1, len(lst) + 1):
            for comb in combinations(lst, r):
                result.append(' '.join(comb))
        return result

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

    # flatten atom & name lists to individ. groups
    group_list = flatten(nested=hit_at_wash, except_last=True)
    name_list = []
    for i, name in enumerate(hit_fg_name_wash):
        for j, individual_group in enumerate(hit_at_wash[i]):
            name_list.append(name+'_'+str(j))

    return group_list, name_list

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
    atom_rings = [list(ring) for ring in atom_rings]
    found_rings = []

    if not atom_rings:
        return [], []

    count_dict = {}
    for i, ring in enumerate(atom_rings):
        matched = False
        try: 
            ring_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=ring, kekuleSmiles=True)
            ring_mol = Chem.MolFromSmiles(ring_smiles)
            for name, pattern in ring_systems_dict.items():
                pattern_mol = Chem.MolFromSmiles(pattern)
                if ring_mol.HasSubstructMatch(pattern_mol):
                    matched = True
                    if name not in count_dict:
                        count_dict[name] = 0
                    else:
                        count_dict[name] += 1
                    found_rings.append(name+'_'+str(count_dict[name]))
                    break
            if not matched:
                uk_label = 'UnknownRing'
                if uk_label not in count_dict:
                    count_dict[uk_label] = 0
                else:
                    count_dict[uk_label] += 1
                found_rings.append(uk_label+'_'+str(count_dict[uk_label]))
        
        except Chem.AtomValenceException as e:
            print(f"AtomValenceException occurred while processing ring {i}: {e}")
        
        except Chem.KekulizeException as e:
            print(f"KekulizeException occurred while processing ring {i}: {e}")
            draw_molecule(f"kek_err_{mol}.png", 'none', highlight_atoms=flatten(ring), color=(200.0/255.0, 40.0/255.0, 20.0/255.0), mol=mol)

        except Exception as e:
            print(f"An unexpected error occurred while processing ring {i}: {e}")
    
    return atom_rings, found_rings

def SME(loaded_model, orig_data, reference, scaler=None, best_worst_activs=None):
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
	    "Piperidine": "C1CCNCC1",
        "Pyridine": "C1=CC=NC=C1",
        "Pyrrole": "C1=CNC=C1",
        "Furan": "C1=COC=C1",
        "Thiophene": "C1=CSC=C1",
        "Imidazole": "C1=CN=CN1",
        "Oxazole": "C1=COC=N1",
        "Thiazole": "C1=NC=NN1",
        "Indole": "C1=CC=C2C(=C1)C=CN2",
	    "Dioxane": "C1CCOOC1.C1CCOOC1",
        "Isoquinoline": "C1=CC=C2C=NC=CC2=C1",
        "Quinoline": "C1=CC=C2C(=C1)C=CC=N2",
        "Naphthalene": "C1=CC=C2C=CC=CC2=C1",
        "Anthracene": "C1=CC=C2C=C3C=CC=CC3=CC2=C1",
        "Phenanthrene": "C1=CC=C2C(=C1)C=CC3=CC=CC=C32"
        # temp dataset
    }

    if best_worst_activs:
        corr_dict, anticorr_dict = best_worst_activs

    mol_dict = {}

    loaded_model.eval()
    for batch, (a, b, e, (y, ID)) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Making Mask-dict", file=sys.stdout, mininterval=10.0):
        ID = ID[0]
        a, b, e = a.to(device), b.to(device), e.to(device)
        if 'ZINC' in ID:
            smile = dataset.smiles[batch]
            ID = smile

        # get functional groups
        mol = Chem.MolFromSmiles(ID)
        fg_at, fg_name = return_fg_hit_atom(mol, fg_name_list, fg_with_ca_list, fg_without_ca_list)
        # print("Hit atoms:", fg_at, "\n Hit names:", fg_name)
        # draw_molecule("test_FGs.png", "ClCCC#N", highlight_atoms=flatten(hit_fg_at), color=(200.0/255.0, 40.0/255.0, 20.0/255.0))

        # get rings
        ring_ats, ring_names = get_ring_names(mol, ring_systems)
        # print("Ring ats:", ring_ats, "Names:\n", ring_names)
        # draw_molecule(f"testr_{ring_names}.png", ID[0], highlight_atoms=flatten(ring_ats), color=(200.0/255.0, 40.0/255.0, 20.0/255.0))

        # scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # scaffold_atom_indices = [atom.GetIdx() for atom in scaffold.GetAtoms()]
        # print("SCFF", scaffold_atom_indices)
        # draw_molecule(f"test_plain.png", smiles)
        # draw_molecule(f"test_rings.png", smiles, highlight_atoms=flatten(atominfo), color=(100.0/255.0, 200.0/255.0, 20.0/255.0))

        # get most anti/corr if provided
        anti_and_corr = []
        anti_and_corr_ats = []
        if best_worst_activs:
            # print('ID:', ID)
            (c_core_at, c_tensor, c_deg) = corr_dict[ID]
            (a_core_at, a_tensor, a_deg) = anticorr_dict[ID]
            corr_ats = get_atom_neighborhood(smile=[ID], center_atom_i=c_core_at, max_degree = c_deg)
            anticorr_ats = get_atom_neighborhood(smile=[ID], center_atom_i=a_core_at, max_degree = a_deg)
            anti_and_corr_ats += [corr_ats] + [anticorr_ats]
            anti_and_corr.append('most_corr')
            anti_and_corr.append('most_anticorr')

        subgr_at = fg_at + ring_ats + anti_and_corr_ats
        subgr_nm = fg_name + ring_names + anti_and_corr
        full_subgr_at = all_combinations(subgr_at, lists=True)
        full_subgr_nm = all_combinations(subgr_nm)
        # print("full subgrat", full_subgr_at)
        # print("full subgr-name", full_subgr_nm)
        # print("Groups:", len(subgr_at), "-(choose)->", len(full_subgr_at))

        # Test every fg-combo group
        masked_a = a.repeat(1+len(full_subgr_at),1,1)  # copy the smile n times
        masked_b = b.repeat(1+len(full_subgr_at),1,1,1)
        masked_e = e.repeat(1+len(full_subgr_at),1,1)

        for copy_i, atom_group in enumerate(full_subgr_at, start=1):
            remove_nodes(atom_group, copy_i, masked_a, masked_b, masked_e)

        with torch.no_grad():
            preds = loaded_model((masked_a, masked_b, masked_e)).reshape(-1, 1)
            if scaler:
                preds = scaler.inverse_transform(preds)
                # print("Scaled pred:", preds)

        # old - new = difference
        original = preds[0]
        preds[1:] = preds[1:] - original

        subgr_changes = defaultdict(float) # may add atoms:{'change': 0.0, 'atoms':[]}
        for groups, change in zip(full_subgr_nm, preds[1:]):
            subgr_changes[groups] = change[0]

        mol_dict[ID] = subgr_changes
    
    return mol_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--d', type=str, required=True)
    args = parser.parse_args()
    data_name = args.d
    # data_name = 'dock_acease_pruned.txt'

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

    # import model
    model_path = find_item_with_keywords('./src/trainingJobs', [args.d, 'model'], dir=False, file=True)
    model_path = model_path[0]
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = dockingProtocol(params=checkpoint['params'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # import activation_dicts (optional)
    most_anticorr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'worst', 'pkl'], dir=False, file=True)
    most_corr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'best', 'pkl'], dir=False, file=True)
    print("Most Corr Path:", most_anticorr_path)
    print("Most Anti Path:", most_corr_path)
    with open(most_anticorr_path[0], 'rb') as file:
        most_anticorr_dict = pickle.load(file)
    with open(most_corr_path[0], 'rb') as file:
        most_corr_dict = pickle.load(file)

    print("Data:", data_name)
    print("Model:", model_path)
    mol_dict = SME(model, data_name, smileData, checkpoint['scaler'], (most_corr_dict, most_anticorr_dict))

    # save results
    output_dir = os.path.join(os.getcwd(), 'results', data_name)
    os.makedirs(output_dir, exist_ok=True)

    dict_path = os.path.join(output_dir, f'{data_name}_sb_mask_dict.pkl')

    with open(dict_path, 'wb') as file:
        pickle.dump(mol_dict, file)

    # print results
    print("------------------\n", len(mol_dict), "mol dict produced.")
    for i,(mol,subgr_dict) in enumerate(mol_dict.items()):
        if i > 20: break
        print("Mol: ", mol)
        for fg,change in subgr_dict.items():
            print("      ", fg, "-- ",change)