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
import pickle
from subgraphs import setup_dataset, get_atom_neighborhood, draw_molecule_with_highlights
from collections import defaultdict
from tqdm import tqdm

def get_numeric_val(lst):
    for elem in lst:
        stripped = elem.translate(elem.maketrans('', '', '. -'))
        if stripped.isdigit():
            return float(elem)
    return None


def plot_lone_gr_vals(mask_dict, out_path, optimize_for_high=True, highlight_groups=None, threshold=3):
    lone_group_dict = defaultdict(lambda: {'count': 0, 'sum': 0.0})

    def search_list_for(search_terms, search_list):
        return [item1 for item1 in search_list if any(item2 in item1 for item2 in search_terms)]

    # Spectral clustering has no intrinsic order. 
    # E.g. Can correctly pick very positive & negative groups, but fail to order them consistently (ex: gr0-,gr1+ in mol1; gr0+,gr1- in mol2)
    # Would avg. to gr0 as 'mid', incorrectly. To avoid this, reorder in ascending order. Introduces slight bias towards method, counterable by keeping this in mind when looking at graphs.
    grad_switch = any(search_list_for(['grad'], highlight_groups))
    if grad_switch:
        for mol, mol_dict in mask_dict.items():
            # Sort, mark old for deletion
            grad_names = []
            grad_vals = []
            keys_to_delete = []  # can't change dict-size while iterating
            
            for subgr, change in mol_dict.items():
                if 'grad' in subgr:
                    grad_vals.append(change.item())
                    grad_names.append(re.sub(r'_\d+$', '', subgr)) # strip '_10' group label
                    keys_to_delete.append(subgr)
            
            for key in keys_to_delete:
                del mol_dict[key]

            paired = sorted(zip(grad_vals, grad_names))
            grad_vals, grad_names = zip(*paired)
            grad_vals, grad_names = list(grad_vals), list(grad_names)

            # Rename
            count = {}
            for i, name in enumerate(grad_names):
                if name in count:
                    grad_names[i] = name + f"_{count[name]}n"
                    count[name] += 1
                else:
                    count[name] = 1
                    grad_names[i] = name + f"_0n"

            # Reinsert
            for grad_val, grad_name in zip(grad_vals, grad_names):
                mol_dict[grad_name] = grad_val
                        

    # store all lone groups info over all mols
    for mol, mol_dict in tqdm(mask_dict.items(), total=len(mask_dict.items()), desc="Making Lone-Group Dict", file=sys.stdout, mininterval=10.0):
        for subgr, change in mol_dict.items():
            groups = subgr.split()
            matches = search_list_for(search_terms=highlight_groups, search_list=groups)
            if (len(groups)==1) and ('Benzene' in groups[0] or '-O' in groups[0]):
                matches = [] # Don't take individual instances of normal lone groups, even if highlighted
            if any(matches):
                for match in matches:
                    if match in lone_group_dict and not ('Benzene' in match or '-O' in match):
                        lone_group_dict[match]['count'] += 1
                        lone_group_dict[match]['sum'] += change
                    else:
                        lone_group_dict[match]['count'] = 1
                        lone_group_dict[match]['sum'] = change
            else:
                # discard non-matching grads/special groups/etc
                if search_list_for(search_terms=['grad', 'corr'], search_list=groups):
                    continue
                # aside from matches, only count lone groups for now. discard numbering, format labels
                if len(groups) != 1: continue
                group = groups[0]
                group = group.split('_') 
                group = [item for item in group if not item.isdigit()]
                if len(group) == 2:
                    group = group[0] + '_' + group[1]
                else: 
                    group = group[0]
                if group in lone_group_dict:
                    lone_group_dict[group]['count'] += 1
                    lone_group_dict[group]['sum'] += change
                else:
                    lone_group_dict[group]['count'] = 1
                    lone_group_dict[group]['sum'] = change

    # rearrange if needed, get avg per non-rare group, sort 
    if optimize_for_high:
        flip_factor = 1
    else:
        flip_factor = -1

    filtered_data = {group: data['sum'] / data['count'] * flip_factor
                    for group, data in lone_group_dict.items() if data['count'] >= threshold}
    sorted_groups = sorted(filtered_data, key=filtered_data.get)
    sorted_changes = [filtered_data[group] for group in sorted_groups]

    # Plot, highlight groups
    plt.figure(figsize=(12, 8))
    color_map = {}
    if highlight_groups:
        colors = plt.get_cmap('tab10').colors
        color_map = {group: colors[i % len(colors)] for i, group in enumerate(highlight_groups)}
        bar_colors = []
        for group in sorted_groups:
            added = False
            for hg in highlight_groups:
                if hg in group: # only color -O, not things containing -O; hacky, replace w fg check later
                    if '-O' in group or 'Benzene' in group:
                        print("Saw one in Group:", group)
                        if '-O' != group and 'Benzene' != group:
                            print("NOT -O or Benz!")
                            break

                    bar_colors.append(color_map.get(hg))
                    added = True
                    break
            if not added:
                bar_colors.append('grey')

        if grad_switch:
            # auto-converts off tensor somewhere in redoing elems
            sorted_changes = [x for x in sorted_changes]
        else:
            sorted_changes = [x.cpu().detach().numpy() for x in sorted_changes]
        # print(sorted_changes)
        bars = plt.bar(sorted_groups, sorted_changes, color=bar_colors)
    else:
        bars = plt.bar(sorted_groups, sorted_changes, color='grey')

    # legend maps highlighted colors
    if highlight_groups:
        handles = [plt.Line2D([0], [0], color=color, lw=4) for group, color in color_map.items()]
        plt.legend(handles, highlight_groups, title='Highlighted Groups')
    plt.xticks(rotation=60, fontsize=10) #45,10
    plt.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--d', type=str, required=True)
    parser.add_argument('-optimize_for_high', '--max', type=bool, help="Set this flag if higher values are better (e.g. solubility). Set false for lower-is-better metrics (e.g. binding affinity).")
    args = parser.parse_args()
    data_name, optimize_for_high = args.d, args.max

    ## Import saved activations
    data_path = find_item_with_keywords('./data', [data_name], dir=False, file=True)
    mask_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'mask', 'pkl'], dir=False, file=True)
    data_path, mask_path = data_path[0], mask_path[0]
    print("Using -- \n Data:", data_name, data_path)
    print("Mask dict:", mask_path)

    # most_anticorr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'worst', 'pkl'], dir=False, file=True)
    # most_corr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'best', 'pkl'], dir=False, file=True)
    # most_anticorr_path, most_corr_path = most_anticorr_path[0], most_corr_path[0]
    # print("Most Corr Path:", most_corr_path)
    # print("Most Anti-corr Path:", most_anticorr_path)

    # with open(most_anticorr_path, 'rb') as file:
    #     most_anticorr_dict = pickle.load(file)
    # with open(most_corr_path, 'rb') as file:
    #     most_corr_dict = pickle.load(file)
    with open(mask_path, 'rb') as file:
        mask_dict = pickle.load(file)

    print("# Mols in Mask Dict:", len(mask_dict))
    for i,(mol,subgr_dict) in enumerate(mask_dict.items()):
        if i > 1: break
        print("Mol: ", mol)
        for fg,change in subgr_dict.items():
            print("      ", fg, "-- ",change)

    o_dir = find_item_with_keywords('./results', [data_name], dir=True, file=False)
    print('odir', o_dir)
    lone_g_opath = os.path.join(o_dir[0], 'lone_groups_hist.png')
    plot_lone_gr_vals(mask_dict, lone_g_opath, optimize_for_high, ['Benzene', '-O', 'grad_suggested'])    

    




