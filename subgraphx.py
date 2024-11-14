import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, AvgPooling
import dgl
from util import *
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from functools import wraps
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
from networkP import dockingProtocol
import math

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data',required=True)
cmdlArgs = parser.parse_args()
data = cmdlArgs.data

NID = "_ID"


class MCTSNode:
    r"""Monte Carlo Tree Search Node

    Parameters
    ----------
    nodes : Tensor
        The node IDs of the graph that are associated with this tree node
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        r"""Get the string representation of the node.

        Returns
        -------
        str
            The string representation of the node
        """
        return str(self.nodes)


class SubgraphX(nn.Module):
    r"""SubgraphX from `On Explainability of Graph Neural Networks via Subgraph
    Explorations <https://arxiv.org/abs/2102.05152>`

    It identifies the most important subgraph from the original graph that
    plays a critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring
    different subgraphs for explanation and uses Shapley values as the measure
    of subgraph importance.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat)`.
        * The output of its forward function is the logits.
    num_hops : int
        Number of message passing layers in the model
    coef : float, optional
        This hyperparameter controls the trade-off between exploration and
        exploitation. A higher value encourages the algorithm to explore
        relatively unvisited nodes. Default: 10.0
    high2low : bool, optional
        If True, it will use the "High2low" strategy for pruning actions,
        expanding children nodes from high degree to low degree when extending
        the children nodes in the search tree. Otherwise, it will use the
        "Low2high" strategy. Default: True
    num_child : int, optional
        This is the number of children nodes to expand when extending the
        children nodes in the search tree. Default: 12
    num_rollouts : int, optional
        This is the number of rollouts for MCTS. Default: 20
    node_min : int, optional
        This is the threshold to define a leaf node based on the number of
        nodes in a subgraph. Default: 3
    shapley_steps : int, optional
        This is the number of steps for Monte Carlo sampling in estimating
        Shapley values. Default: 100
    log : bool, optional
        If True, it will log the progress. Default: False
    """

    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=True,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log
        self.model_in = None
        self.na = 0

        self.model = model

    def shapley(self, subgraph_nodes):
        r"""Compute Shapley value with Monte Carlo approximation.

        Parameters
        ----------
        subgraph_nodes : tensor
            The tensor node ids of the subgraph that are associated with this
            tree node

        Returns
        -------
        float
            Shapley value
        """
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                exclude_feat = T.from_numpy(padDim(exclude_feat.numpy(), 70, dim=0)).float().unsqueeze(0)
                exclude_probs = self.model(
                    (exclude_feat, self.model_in[1], self.model_in[2])
                ) # no softmax (dim = -1)
                exclude_value = exclude_probs

                include_feat = T.from_numpy(padDim(include_feat.numpy(), 70, dim=0)).float().unsqueeze(0)
                include_probs = self.model(
                   (include_feat, self.model_in[1], self.model_in[2])
                ) # no softmax (dim = -1)
                include_value = include_probs
                
            marginal_contributions.append(include_value - exclude_value)

        return T.Tensor(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = dgl.subgraph.node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = dgl.transforms.functional.remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = dgl.to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        r"""Perform a MCTS rollout.

        Parameters
        ----------
        mcts_node : MCTSNode
            Starting node for MCTS

        Returns
        -------
        float
            Reward for visiting the node this time
        """
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, na, model_in, target_class, **kwargs):
        self.model.eval()
        assert (
            graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}." 

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs
        self.model_in = model_in
        self.na = na

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root = MCTSNode(graph.nodes())
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return best_leaf.nodes

smileData = pd.read_csv('./data/smilesDS.smi', delimiter=' ')
smileData.columns = ['smiles', 'zinc_id']
smileData.set_index('zinc_id', inplace=True)

data_path = f'./data/{data}.txt'
allData = labelsToDF(data_path)
if 'smiles' not in allData.columns:
    allData = pd.merge(allData, smileData, on='zinc_id')

trainData, valData, testData = np.split(allData.sample(frac=1), 
                                        [int(.70*len(allData)), int(.85*len(allData))])

ID_column = "Compound_ID"

yTrain = trainData['labels'].values
xTest = testData[[ID_column, 'smiles']].values.tolist()
yTest = testData['labels'].values

scaler = StandardScaler()
yTrain = yTrain.reshape(-1, 1)
yTrain = scaler.fit_transform(yTrain).T[0].tolist()  
yTest = scaler.transform(yTest.reshape(-1, 1)).T[0].tolist()

modeld = torch.load("./src/model1/r_sol_data_ESOL_model.pth")
model = dockingProtocol(modeld["params"]).to(device)
print(testData['labels'].values.mean(), xTest[0][1])
m = MolFromSmiles(xTest[0][1])
mol_img = Draw.MolToImage(m)
mol_img.save('test_mol.png')
a, b, e = buildFeats([xTest[0][1]])
na, nb = np.count_nonzero(a.squeeze().sum(axis=1)), np.count_nonzero(e.squeeze().flatten() != -1) // 2 
print("num atoms", na, "num bonds", nb)
# print(a.shape, b.shape, e.shape)
a1, b1, e1 = a.squeeze().numpy()[:na, :], b.squeeze().numpy(), e.squeeze().numpy()[:nb, :]
e_id = np.where(e1 != -1)
edge_index = np.stack((e_id[0], e1[e_id]), axis=0)
edge_attr = b1[(e_id[0], e_id[1])]

data = torch_geometric.data.Data(x=torch.from_numpy(a1), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr))
# g = torch_geometric.utils.to_networkx(data, to_undirected=True)
# pos = nx.spring_layout(g)
# nx.draw(g, pos)
# plt.show()
g = torch_geometric.utils.to_dgl(data)

# data0 = GINDataset('MUTAG', self_loop=True)
# dataloader0 = GraphDataLoader(data, batch_size=64, shuffle=True)
# g0, l0 = data0[0]
# graph_feat0 = g0.ndata.pop("attr")

explainer = SubgraphX(model, num_hops=6)
graph_feat = g.ndata.pop('x')
g_nodes_explain = explainer.explain_graph(g, graph_feat,
                                          na,
                                          (a, b, e),
                                          target_class=0)


# dataset_hf = load_dataset("graphs-datasets/<mydataset>")
# # For the train set (replace by valid or test as needed)
# dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]
# dataset_pg = DataLoader(dataset_pg_list)