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

    # store all lone groups info over all mols
    for mol, mol_dict in tqdm(mask_dict.items(), total=len(mask_dict.items()), desc="Making Lone-Group Dict", file=sys.stdout, mininterval=10.0):
        for subgr, change in mol_dict.items():
            groups = subgr.split()
            if len(groups) != 1: continue
            # for lone groups: discard numbering, format labels
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
    if not optimize_for_high:
        flip_factor = -1
    else:
        flip_factor = 1

    # print("Optimized for high:", optimize_for_high)
    filtered_data = {group: data['sum'] / data['count'] * flip_factor
                    for group, data in lone_group_dict.items() if data['count'] >= threshold}
    # print("filtered data", filtered_data)
    sorted_groups = sorted(filtered_data, key=filtered_data.get)
    sorted_changes = [filtered_data[group] for group in sorted_groups]
    # print("sorted groups", len(sorted_groups), sorted_groups)
    # print("sorted changes", len(sorted_changes), sorted_changes)

    # Plot, highlight groups
    plt.figure(figsize=(12, 8))
    color_map = {}
    if highlight_groups:
        colors = plt.get_cmap('tab10').colors
        color_map = {group: colors[i % len(colors)] for i, group in enumerate(highlight_groups)}
        bar_colors = [color_map.get(group, 'grey') for group in sorted_groups]
        sorted_changes = [x.cpu().detach().numpy() for x in sorted_changes]
        print(sorted_changes)
        bars = plt.bar(sorted_groups, sorted_changes, color=bar_colors)
    else:
        bars = plt.bar(sorted_groups, sorted_changes, color='grey')

    # legend maps highlighted colors
    if highlight_groups:
        handles = [plt.Line2D([0], [0], color=color, lw=4) for group, color in color_map.items()]
        plt.legend(handles, highlight_groups, title='Highlighted Groups')
    plt.xticks(rotation=45)
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
    # most_anticorr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'worst', 'pkl'], dir=False, file=True)
    # most_corr_path = find_item_with_keywords(f'./results/{data_name}', [data_name, 'best', 'pkl'], dir=False, file=True)
    print("Using -- \n Data:", data_name, data_path)
    print("Mask dict:", mask_path)
    # print("Most Corr Path:", most_corr_path)
    # print("Most Anti-corr Path:", most_anticorr_path)
    # data_path, mask_path, most_anticorr_path, most_corr_path = data_path[0], mask_path[0], most_anticorr_path[0], most_corr_path[0]
    data_path, mask_path = data_path[0], mask_path[0]

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
    plot_lone_gr_vals(mask_dict, lone_g_opath, optimize_for_high, ['Benzene'])    

    




