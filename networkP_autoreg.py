import torch
import torch.nn as nn
from features import \
    num_atom_features, \
    num_bond_features
import numpy as np
from util import buildFeats
from util import dockingDataset
import torch.nn.functional as F
import os
from features import *

from networkP import GraphLookup, nfpConv, nfpOutput, GraphPool


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")
print(f'num interop threads: {torch.get_num_interop_threads()}, num intraop threads: {torch.get_num_threads()}') 


class SubgraphSum(nn.Module):
    """Sum/pool node embeddings into a single subgraph embedding"""
    def __init__(self):
        super(SubgraphSum, self).__init__()

    def forward(self, activations):
        embeds_sum = torch.sum(activations, dim=1)
        return embeds_sum


class GCN(nn.Module):
    def __init__(self, layers, fpl=32, hf=32):
        super(GCN, self).__init__()
        self.layers = layers
        self.fpl = fpl
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.layersArr, self.outputArr = self.init_layers()
        self.op = nfpOutput(self.layers[-1], self.fpl)
        self.pool = GraphPool()
        self.subgraph_sum = SubgraphSum()
        self.to(device)

    def init_layers(self):
        layersArr, outputArr = [], []
        i_size = num_atom_features(just_structure=True)

        for idx, (i, o) in enumerate(self.throughShape):
            outputArr.append(nfpOutput(self.layers[idx], self.fpl))
            layersArr.append(nfpConv(i, o, just_structure=True))
        outputArr.append(nfpOutput(self.layers[-1], self.fpl))
        # print("FPL when initing:", self.fpl)
        # print("Layers Arr:", nn.ModuleList(layersArr))
        # print("Output Arr:", nn.ModuleList(outputArr))
        return nn.ModuleList(layersArr), nn.ModuleList(outputArr)
    
    def forward(self, input, idx_list=None):
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        lay_count = len(self.layers[1:])
        for i in range(lay_count):
            a = self.layersArr[i]((a, b, e)) # calls nfpConv layer on inputs
            # print(f"Layer {i}: {a.shape}")
            a = self.pool(a, e)
            # print(f"-pool->{a.shape}")
        subgraph_embedding = self.subgraph_sum(a)
        # print(f"SBgraph shape: {subgraph_embedding.shape}")
        # print(f"Nodes shape: {a.shape}")

        if idx_list is None:
            return subgraph_embedding
        else:
            node_embeds_list = []
            for i in idx_list:
                node_embeds_list.append(a[:,i,:])

            return subgraph_embedding, node_embeds_list

class MLP(nn.Module):
    def __init__(self, in_size, out_size, dropout, ba, out_type='ReLU'):
        super(MLP, self).__init__()
        self.i = in_size
        self.o = out_size
        self.ba = ba
        self.arch = None
        self.dropout = dropout
        self.mlp = nn.Sequential()
        self.buildModel(self.i,self.o, out_type)
    
    def buildModel(self, in_size, out_size, out_type='None'):
        self.ba = [int(round(l * in_size)) for l in self.ba] # make layers porportional to input size
        self.arch = [(in_size, self.ba[0])] + list(zip(self.ba[:-1], self.ba[1:])) + [(self.ba[-1], out_size)] 
        for j, (i, o) in enumerate(self.arch):
            # print(f"Lay {j}: {i}->{o}")
            self.mlp.add_module(f'relu act {j}', nn.ReLU())
            self.mlp.add_module(f'layer norm {j}', nn.LayerNorm(i)) #since batch size drops as molecules are completed
            self.mlp.add_module(f'dropout {j}', nn.Dropout(self.dropout))
            self.mlp.add_module(f'linear {j}', nn.Linear(i, o))
            nn.init.constant_(self.mlp[-1].bias, .03)
        if out_type == 'ReLU':
            self.mlp.add_module(f'final relu', nn.ReLU())
        if out_type == 'softplus':
            self.mlp.add_module(f'final softplus', nn.Softplus())
 
    def forward(self, embeddings):
        i_size = embeddings.shape[1]
        if self.arch is None:
            self.buildModel(i_size, self.o_size)
        return self.mlp(embeddings)

    
class GCN_Autoreg(nn.Module):
    def __init__(self, params):
        super(GCN_Autoreg, self).__init__()
        self.node_toggle = True
        self.GCN = GCN(
                layers=params["conv"]["layers"],
                fpl= params["fpl"]
            )
        self.Node_Pred = MLP(in_size=64, out_size=43, dropout=.2, ba=[1,1])
        self.Edge_Pred = MLP(in_size=192, out_size=4, dropout=.2, ba=[1,1])
        self.to(device)

    def forward(self, a_b_e_input, pred_node=True, idx_orig=None, idx_dest=None):
        self.node_toggle = pred_node
        if pred_node:
            subgr_embeds = self.GCN(a_b_e_input)
            n_feats = num_atom_features(just_structure=True)

            pred = self.Node_Pred(subgr_embeds)
        
        else: # predict edge/bond between two arbitrary nodes
            subgr_embeds, [orig_embed, dest_embed] = self.GCN(a_b_e_input, [idx_orig, idx_dest])

            combined = torch.cat((subgr_embeds, orig_embed, dest_embed), axis=1)
            n_feats = num_bond_features(just_structure=True)
            pred = self.Edge_Pred(combined)
        
        return pred
    
    def save(self, params, outpath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'params': params,
        }, outpath)