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

class AtomPositionalEmbedding(nn.Module):
    def __init__(self, max_atoms=70, embed_dim=49): # assumes 'no structure'
        super(AtomPositionalEmbedding, self).__init__()
        self.pos_embed = nn.Embedding(max_atoms, embed_dim)
        self.pos_embed.weight.requires_grad = True
        self._initialize_padding_embedding()

    def _initialize_padding_embedding(self):
        with torch.no_grad():
            self.pos_embed.weight[0] = torch.zeros(self.pos_embed.embedding_dim)
            self.pos_embed.weight[0].detach().requires_grad = False  # freeze padding atom embeddings

    def forward(self, atom_matrix):
        present_atom_mask = ~(atom_matrix.sum(dim=2) == 0)  # mask to identify non-padding atoms
        present_atom_mask = torch.flip(present_atom_mask, dims=[1]).int()
        distance_vectors = torch.cumsum(present_atom_mask, dim=1)
        distance_vectors = torch.flip(distance_vectors, dims=[1])  # distance from final node

        pos_embeds = self.pos_embed(distance_vectors)
        return pos_embeds

class GCN(nn.Module):
    def __init__(self, layers, fpl=32, hf=32, max_atoms=70, embed_dim=32):
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
        return nn.ModuleList(layersArr), nn.ModuleList(outputArr)

    def forward(self, input, idx_list=None):
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)

        lay_count = len(self.layers[1:])
        skip_conn = None
        for i in range(lay_count):
            a = self.layersArr[i]((a, b, e))
            a = self.pool(a, e)
            if i == 0:
                skip_conn = self.subgraph_sum(a)
        subgraph_embedding = self.subgraph_sum(a)
        subgraph_embedding = subgraph_embedding + skip_conn

        if idx_list is None:
            return subgraph_embedding
        else:
            node_embeds_list = [a[:, i, :] for i in idx_list]
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
        self.node_pred_pos = AtomPositionalEmbedding()
        self.edge_pred_pos = AtomPositionalEmbedding()
        self.GCN = GCN(
                layers=params["conv"]["layers"],
                fpl= params["fpl"]
            )
        self.Node_Pred = MLP(in_size=64, out_size=44, dropout=.2, ba=[1,1])
        self.Edge_Pred = MLP(in_size=192, out_size=4, dropout=.2, ba=[1,1])
        self.to(device)

    def forward(self, a_b_e_input, pred_node=True, idx_orig=None, idx_dest=None):
        self.node_toggle = pred_node
        (a,b,e) = a_b_e_input
        if pred_node:
            pos_embeds = self.node_pred_pos(a)
            a = a + pos_embeds
            subgr_embeds = self.GCN((a,b,e))
            n_feats = num_atom_features(just_structure=True)
            pred = self.Node_Pred(subgr_embeds)
        
        else: # predict edge/bond between two arbitrary nodes
            pos_embeds = self.edge_pred_pos(a)
            a = a + pos_embeds
            subgr_embeds, [orig_embed, dest_embed] = self.GCN((a,b,e), [idx_orig, idx_dest])

            combined = torch.cat((subgr_embeds, orig_embed, dest_embed), axis=1)
            n_feats = num_bond_features(just_structure=True)
            pred = self.Edge_Pred(combined)
        
        return pred
    
    def save(self, params, outpath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'params': params,
        }, outpath)