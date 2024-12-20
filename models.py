import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
import torch.optim as optim
import numpy as np
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
import dgl.function as fn
import pyro
from torch_geometric.nn import global_mean_pool
from dgl.nn import GraphConv
from torch_scatter import scatter_mean, scatter_add, scatter_std
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn import Set2Set
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import SAGEConv


class MLP(torch.nn.Module):

    def __init__(self, num_features, num_classes, dims=16):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, dims),
            torch.nn.ReLU(),
            torch.nn.Linear(dims, num_classes))

    def forward(self, x):
        x = self.mlp(x)
        return x


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, learn_eps=False))  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h):

        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
        return h


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.pytorch.conv.GraphConv(num_features, hidden_dim * 2, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.pytorch.conv.GraphConv(hidden_dim * 2, hidden_dim * 2, allow_zero_in_degree=True)
        self.conv3 = dgl.nn.pytorch.conv.GraphConv(hidden_dim * 2, hidden_dim, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat);
        h = F.relu(h)
        h = self.conv2(g, h);
        h = F.relu(h)
        h = self.conv3(g, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat);
        h = F.relu(h)
        h = self.conv2(g, h);
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class Mainmodel_domainadapt(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename,
                 encoder):
        super().__init__()
        self.tau = 1.0
        self.readout = args.readout_f
        self.s2s = Set2Set(hidden_dim, 2, 1)
        self.s2s_rev = Set2Set(in_dim, 2, 1)
        # if args.transfer_mode ==1:
        self.in_dim = args.d_transfer
        # else:
        # 	self.in_dim  = in_dim
        self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)

        self.batch_size = args.batch_size
        self.useAtt = args.useAtt
        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
        self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.num_nodes = -1
        self.device = args.device
        self.r_transfer_d = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, in_dim * 2))

        if args.task == "graph_regression":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
        elif args.task == "graph_classification":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:
            print(f"checking mainmodel_finetuning task ...")

        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = GIN(self.in_dim, hidden_dim)
        elif encoder == "GCN":
            self.Encoder1 = GCN(self.in_dim, hidden_dim)
            self.Encoder2 = GCN(self.in_dim, hidden_dim)
        elif encoder == "GraphSAGE":
            self.Encoder1 = GraphSAGE(self.in_dim, hidden_dim)
            self.Encoder2 = GraphSAGE(self.in_dim, hidden_dim)
        elif encoder == "Transformer":  # in_dim, hidden_dim, num_layers, num_heads, device
            self.Encoder1 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
            self.Encoder2 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()

        print(f"Loading pre-trained model .pt  ... ")
        self.model = torch.load(cp_filename, map_location=args.device)
        for p in self.model.parameters():
            p.requires_grad = True

        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1))
        self.reconstructX = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, in_dim))

    def compress(self, graph_features, device):
        p = self.compressor(graph_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        self.tau = 1
        num_nodes = z1.size(0)  # 32
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
                                                                                                                      i + 1) * batch_size].diag())))
        ret = torch.cat(losses)
        return ret.mean()

    def compression(self, nodes_list, device):
        epsilon = 0.0000001
        noisy_node_feature_all = torch.tensor(()).to(device)
        p_all = torch.tensor(()).to(device)
        KL_tensor_all = torch.tensor(()).to(device)

        z = len(nodes_list)
        graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
        for i in range(z):
            features = graph_feature_split[i]

            lambda_pos, p = self.compress(features, device)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_node_feature = features.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
            noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std
            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
                noisy_node_feature_mean) * noisy_node_feature_std

            noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

            p_all = torch.cat((p_all, p), 0)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
                ((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
            KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
        return noisy_node_feature_all, p_all, KL_tensor_all

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, current_epoch, edge_index,
                k_transition, device, batch_size=16):
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()
        self.device = device
        batch_x_org = batch_x
        batch_x_transfer = self.transfer_d(batch_x)
        x_subs = self.transfer_d(x_subs)

        interaction_map, _, _, _ = self.model.extract_features(nodes_list, batch_g, batch_x_transfer,
                                                               flatten_batch_subgraphs, x_subs, device)

        interaction_map = self.MLP(interaction_map)  # 2d --> d

        # 9. X loss
        interaction_map = self.s2s(batch_g, interaction_map)  # nb 2d
        interaction_map = self.r_transfer_d(interaction_map)  # nb 2in_dim
        org_X = self.s2s_rev(batch_g, batch_x_org)  # nb 2in_dim

        X_loss = self.loss_X(org_X, interaction_map)
        return X_loss

    def loss_X(self, batch_x_org, interaction_map):
        # interaction_map_X = self.reconstructX(interaction_map)
        # loss = F.mse_loss(interaction_map, batch_x_org)
        row_num, col_num = interaction_map.size()
        loss = torch.sum((interaction_map - batch_x_org) ** 2) # / (row_num)
        return loss

    def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, device):
        # 1. GNN encoder for graph batch
        graph_features = self.Encoder1(batch_g, batch_x)

        # 2. Subgraph extractions and Encoder
        subgraphs_features = self.Encoder2(flatten_batch_subgraphs, x_subs)

        # 3. Add normalization
        self.graph_features = graph_features  # F.normalize(graph_features, dim = 1)
        self.subgraphs_features = subgraphs_features  # F.normalize(subgraphs_features, dim = 1)

        # Pooling nb x 2d
        if self.readout == "sum":
            batch_g.ndata['h'] = self.graph_features
            graph_features_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
        else:
            graph_features_readout = self.s2s(batch_g, self.graph_features)  # [nb 2d]

        # 4. Compression p: preserve_rate
        noisy_node_feature, p, KL_tensor = self.compression(nodes_list, device)

        # 5. Core - subgraphs
        flatten_batch_subgraphs.ndata['h'] = self.subgraphs_features
        subgraphs_features_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'h')
        interaction_map = torch.cat((noisy_node_feature, subgraphs_features_readout), -1)

        # Attention-based interaction
        if self.useAtt:
            subgs_att = torch.tensor(()).to(device)
            if self.readout == "sum":
                batch_g.ndata['h'] = noisy_node_feature
                noisy_node_feature_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
            else:
                noisy_node_feature_readout = self.s2s(batch_g, noisy_node_feature)  # nb x 2d
                noisy_node_feature_readout = self.reduce_d(noisy_node_feature_readout)

            subgraphs_features_readout_split = torch.split(subgraphs_features_readout, tuple(nodes_list))
            z = len(subgraphs_features_readout_split)
            for i in range(z):
                noisy_node_feature_readout_cp = noisy_node_feature_readout[i].repeat(nodes_list[i], 1)
                interaction = torch.cat((noisy_node_feature_readout_cp, subgraphs_features_readout_split[i]),
                                        -1)  # x[n, d]     [n, d] -->  [n  2d]

                layer_atten = self.attn_layer(interaction)
                layer_atten = F.softmax(layer_atten, dim= 0)
                a = subgraphs_features_readout_split[i] * layer_atten  # [n , d]
                subgs_att = torch.cat((subgs_att, a), 0)
            interaction_map = torch.cat((noisy_node_feature, subgs_att), -1)
        return interaction_map, KL_tensor, noisy_node_feature, graph_features_readout

    #######################################################		end shared pre-training

    def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
        row_num, col_num = interaction_map.size()
        adj = org_graph.adj().to_dense()
        recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

        loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num)
        return loss

    def loss_recon(self, interaction_map, trans_logM, nodes_list):

        sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
        loss = 0
        z = len(nodes_list)
        for k in range(z):
            h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
            row_num, col_num = h.size()
            for i in range(self.k_transition):
                loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
                            row_num * col_num)
        loss = loss / (self.k_transition)
        return loss


class Mainmodel_finetuning(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename,
                 encoder):
        super().__init__()
        self.tau = 1.0
        self.dataset =args.dataset
        self.readout = args.readout_f
        self.s2s = Set2Set(hidden_dim, 2, 1)

        # if args.transfer_mode ==1:
        self.in_dim = args.d_transfer
        # else:
        # 	self.in_dim  = in_dim
        self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)

        self.batch_size = args.batch_size
        self.useAtt = args.useAtt
        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
        self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.num_nodes = -1
        self.device = args.device
        self.dataset = args.dataset
        self.tasks = ['ZINC', 'Peptides-struct', 'FreeSolv', 'ESOL']
        if args.task == "graph_regression":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
        elif args.task == "graph_classification":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:
            print(f"checking mainmodel_finetuning task ...")

        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = GIN(self.in_dim, hidden_dim)
        elif encoder == "GCN":
            self.Encoder1 = GCN(self.in_dim, hidden_dim)
            self.Encoder2 = GCN(self.in_dim, hidden_dim)
        elif encoder == "GraphSAGE":
            self.Encoder1 = GraphSAGE(self.in_dim, hidden_dim)
            self.Encoder2 = GraphSAGE(self.in_dim, hidden_dim)
        elif encoder == "Transformer":  # in_dim, hidden_dim, num_layers, num_heads, device
            self.Encoder1 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
            self.Encoder2 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()

        print(f"Loading pre-trained model .pt  ... ")
        self.model = torch.load(cp_filename, map_location=args.device)

        for p in self.model.parameters():
            p.requires_grad = False
        num_layers = 4
        unfrezz_layers = ["layers." + str(num_layers), "layers." + str(num_layers - 1) , "layers." + str(num_layers - 2)]
        unfrezz_batch_norms = ["batch_norms." + str(num_layers), "batch_norms." + str(num_layers - 1) ]
        for name, para in self.model.named_parameters():
            for layer in unfrezz_layers:
                if layer in name :
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        #self.unfree_layers(unfrezz_layers, unfrezz_batch_norms)
        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1))
    def unfree_layers(self, unfrezz_layers, unfrezz_batch_norms):
        for name, para in self.model.named_parameters():
            for layer in unfrezz_layers:
                if layer in name :
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        # for name, para in self.model.named_parameters():
        #     for layer in unfrezz_batch_norms:
        #         if layer in name :
        #             para.requires_grad = True
        #         else:
        #             para.requires_grad = False
    def compress(self, graph_features, device):
        p = self.compressor(graph_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def compression(self, nodes_list, device):
        epsilon = 0.0000001
        noisy_node_feature_all = torch.tensor(()).to(device)
        p_all = torch.tensor(()).to(device)
        KL_tensor_all = torch.tensor(()).to(device)

        z = len(nodes_list)
        graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
        for i in range(z):
            features = graph_feature_split[i]

            lambda_pos, p = self.compress(features, device)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_node_feature = features.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
            noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std
            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
                noisy_node_feature_mean) * noisy_node_feature_std

            noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

            p_all = torch.cat((p_all, p), 0)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
                ((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
            KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
        return noisy_node_feature_all, p_all, KL_tensor_all

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, x_subs, current_epoch, edge_index, k_transition,
                device, batch_size=2):
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()
        self.device = device

        batch_x = self.transfer_d(batch_x)
        x_subs = self.transfer_d(x_subs)

        interaction_map, _, _, _ = self.model.extract_features(nodes_list, batch_g, batch_x, flatten_batch_subgraphs,
                                                               x_subs, device)  # n 2d

        interaction_map = self.MLP(interaction_map)  # 2d ---> d

        interaction_map = self.s2s(batch_g, interaction_map)  # nb 2d
        if self.dataset in self.tasks:
            return self.predict(interaction_map), 0, 0, 0
        else:
            sig = nn.Sigmoid()
            return sig(self.predict(interaction_map)), 0, 0, 0

    def loss(self, scores, targets):
        loss = nn.BCELoss()
        l = loss(scores.float(), targets.float())
        return l

    def loss_CrossEntropy(self, scores, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores.to(torch.float32), targets.squeeze(dim=-1))
        return loss

    def loss_RMSE(self, scores, targets):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(scores, targets))
        return loss
    def BCEWithLogitsLoss(self, scores, targets):
        loss = nn.BCEWithLogitsLoss()(scores, targets)

        return loss

    def lossMAE(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss


class Mainmodel(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, encoder):
        super().__init__()
        self.tau = 1.0
        self.recons_type = args.recons_type
        self.useAtt = args.useAtt

        self.readout = args.readout_f
        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
        self.fc1 = torch.nn.Linear(hidden_dim, 1)

        self.in_dim = args.d_transfer
        self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)

        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)
        self.attn_layer = nn.Linear(self.hidden_dim * 2, 1)
        self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.device = args.device
        self.s2s = Set2Set(hidden_dim, 2, 1)

        self.reconstructX = nn.Sequential(
            nn.Linear(self.hidden_dim, self.in_dim))
        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = GIN(self.in_dim, hidden_dim)
        elif encoder == "GCN":
            self.Encoder1 = GCN(self.in_dim, hidden_dim)
            self.Encoder2 = GCN(self.in_dim, hidden_dim)
        elif encoder == "GraphSAGE":
            self.Encoder1 = GraphSAGE(self.in_dim, hidden_dim)
            self.Encoder2 = GraphSAGE(self.in_dim, hidden_dim)
        elif encoder == "Transformer":
            self.Encoder1 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
            self.Encoder2 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()

        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1))

    def compress(self, graph_features, device):
        p = self.compressor(graph_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        self.tau = 1
        num_nodes = z1.size(0)  # 32
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
                                                                                                                      i + 1) * batch_size].diag())))
        ret = torch.cat(losses)
        return ret.mean()

    def compression(self, nodes_list, device):
        epsilon = 0.0000001
        noisy_node_feature_all = torch.tensor(()).to(device)
        p_all = torch.tensor(()).to(device)
        KL_tensor_all = torch.tensor(()).to(device)

        z = len(nodes_list)
        graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
        for i in range(z):
            features = graph_feature_split[i]

            lambda_pos, p = self.compress(features, device)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_node_feature = features.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
            noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std
            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
                noisy_node_feature_mean) * noisy_node_feature_std

            noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

            p_all = torch.cat((p_all, p), 0)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
                ((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
            KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
        return noisy_node_feature_all, p_all, KL_tensor_all

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, current_epoch, edge_index,
                k_transition, device, batch_size=16):
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()
        self.device = device

        batch_x = self.transfer_d(batch_x)
        x_subs = self.transfer_d(x_subs)

        interaction_map, KL_tensor, noisy_node_feature, graph_features_readout = self.extract_features(nodes_list,
                                                                                                       batch_g, batch_x,
                                                                                                       flatten_batch_subgraphs,
                                                                                                       x_subs, device)

        interaction_map = self.MLP(interaction_map)

        # 6. KL upper bound
        KL_Loss = torch.mean(KL_tensor)

        # 7. Contrastive loss
        if self.readout == "sum":
            batch_g.ndata['h'] = noisy_node_feature
            noisy_node_feature_2 = dgl.sum_nodes(batch_g, 'h')  # [nb d]
        else:
            noisy_node_feature_2 = self.s2s(batch_g, noisy_node_feature)  # [nb , 2d]
        contrastive_loss = self.batched_semi_loss(noisy_node_feature_2, graph_features_readout, self.batch_size)

        # 8. Reconstruction loss
        if self.recons_type == 'adj':
            reconstruction_loss = self.loss_recon_adj(interaction_map, batch_g)
        elif self.recons_type == 'logM':
            reconstruction_loss = self.loss_recon(interaction_map, batch_logMs, nodes_list)
        else:
            reconstruction_loss = -1.0

        # 9. X loss
        # X_loss = self.loss_X(batch_g, interaction_map)
        # reconstruction_loss+= X_loss
        return None, KL_Loss, contrastive_loss, reconstruction_loss

    def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, device):
        # 1. GNN encoder for graph batch
        graph_features = self.Encoder1(batch_g, batch_x)

        # 2. Subgraph extractions and Encoder
        subgraphs_features = self.Encoder2(flatten_batch_subgraphs, x_subs)

        # 3. Add normalization
        self.graph_features = graph_features  # F.normalize(graph_features, dim = 1)
        self.subgraphs_features = subgraphs_features  # F.normalize(subgraphs_features, dim = 1)

        # Pooling nb x 2d
        if self.readout == "sum":
            batch_g.ndata['h'] = self.graph_features
            graph_features_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
        else:
            graph_features_readout = self.s2s(batch_g, self.graph_features)  # [nb 2d]

        # 4. Compression p: preserve_rate
        noisy_node_feature, p, KL_tensor = self.compression(nodes_list, device)

        # 5. Core - subgraphs
        flatten_batch_subgraphs.ndata['h'] = self.subgraphs_features
        subgraphs_features_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'h')
        interaction_map = torch.cat((noisy_node_feature, subgraphs_features_readout), -1)

        # Attention-based interaction
        if self.useAtt:
            subgs_att = torch.tensor(()).to(device)
            if self.readout == "sum":
                batch_g.ndata['h'] = noisy_node_feature
                noisy_node_feature_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
            else:
                noisy_node_feature_readout = self.s2s(batch_g, noisy_node_feature)  # nb x 2d
                noisy_node_feature_readout = self.reduce_d(noisy_node_feature_readout)

            subgraphs_features_readout_split = torch.split(subgraphs_features_readout, tuple(nodes_list))
            z = len(subgraphs_features_readout_split)
            for i in range(z):
                noisy_node_feature_readout_cp = noisy_node_feature_readout[i].repeat(nodes_list[i], 1)
                interaction = torch.cat((noisy_node_feature_readout_cp, subgraphs_features_readout_split[i]),
                                        -1)  # x[n, d]     [n, d] -->  [n  2d]

                layer_atten = self.attn_layer(interaction)
                layer_atten = F.softmax(layer_atten, dim=0)
                a = subgraphs_features_readout_split[i] * layer_atten  # [n , d]
                subgs_att = torch.cat((subgs_att, a), 0)
            interaction_map = torch.cat((noisy_node_feature, subgs_att), -1)
        return interaction_map, KL_tensor, noisy_node_feature, graph_features_readout

    #######################################################		end shared pre-training aaaaa aaa aa  aa
    def loss(self, scores, targets):
        loss = nn.BCELoss()
        l = loss(scores.float(), targets.float())
        return l
    def loss_X(self, batch_g, interaction_map):
        interaction_map_X = self.reconstructX(interaction_map)
        loss = F.mse_loss(interaction_map_X, batch_g.ndata['x'])
        return loss

    def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
        row_num, col_num = interaction_map.size()
        adj = org_graph.adj().to_dense()
        recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

        loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num)
        return loss

    def loss_recon(self, interaction_map, trans_logM, nodes_list):

        sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
        loss = 0
        z = len(nodes_list)
        for k in range(z):
            h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
            row_num, col_num = h.size()
            for i in range(self.k_transition):
                loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
                            row_num * col_num)
        loss = loss / (self.k_transition)
        return loss


class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, inp / 2),
            torch.nn.ReLU(),
            torch.nn.Linear(inp / 2, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x


##############################################################################################

# graph.ndata["h"] = feat
# individual_graphs = dgl.unbatch(graph)
# node_ids = [g.nodes()[g.ndata["filter_attr"].bool()] for g in individual_graphs]
# all_subgraphs = [dgl.node_subgraph(g, ids) for g, ids in zip(individual_graphs, node_ids)]
# batched_subgraphs = dgl.batch(all_subgraphs)

class Transformer(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, device):
        super().__init__()
        self.h = None
        self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_heads))

    def extract_features(self, g, h):
        h = self.embedding_h(h)
        for layer in self.layers:
            h = layer(h, g)
        return h

    def forward(self, g, h):
        h = self.extract_features(g, h)
        return h


class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads)

        self.O = nn.Linear(out_dim, out_dim)

        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, h, g):
        h_in1 = h  # for first residual connection

        attn_out = self.attention(h, g)
        h = attn_out.view(-1, self.out_channels)
        h = self.O(h)
        h = h_in1 + h  # residual connection
        h = self.layer_norm1(h)
        h_in2 = h  # for second residual connection
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)

        h = F.dropout(h, 0.5, training=self.training)
        h = self.FFN_layer2(h)
        h = h_in2 + h  # residual connection
        h = self.layer_norm2(h)
        return h


class MultiHeadAttentionLayer(nn.Module):
    # in_dim, out_dim, num_heads
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        self.hidden_size = in_dim  # 80
        self.num_heads = num_heads  # 8
        self.head_dim = out_dim // num_heads  # 10

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        # scaling scale
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(exp('score'))
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, dgl.function.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # src_mul_edge
        g.send_and_recv(eids, dgl.function.copy_e('score', 'score'), fn.sum('score', 'z'))  # copy_edge

    # Update:
    # If I replace fn.u_mul_e('h', 'edge_weight', 'm') by
    # lambda e: {'m' : e.src['h'] * e.data['edge_weight']}, it seems to solve the problem and give consistent results
    # (h, g, phi,  current_epoch)
    def forward(self, h, g):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        return h_out


#############################################################################################################################################
#############################################################################################################################################

class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(inp, outp))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, dims * 2),
            torch.nn.SELU(),
            torch.nn.Linear(dims * 2, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPA(torch.nn.Module):

    def __init__(self, in_feats, dim_h, dim_z):
        super(MLPA, self).__init__()

        self.gcn_mean = torch.nn.Sequential(
            torch.nn.Linear(in_feats, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_z)
        )

    def forward(self, hidden):
        # GCN encoder
        Z = self.gcn_mean(hidden)
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


#############################################################################################################################################

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)

        return y


"""
	Util functions
"""


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: (((edges.data[field])) / scale_constant)}

    return func


#############################################################################################################################################
#############################################################################################################################################
class Mainmodel_continue(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename,
                 encoder):
        super().__init__()
        self.tau = 1.0
        self.readout = args.readout_f
        self.s2s = Set2Set(hidden_dim, 2, 1)
        self.s2s_rev = Set2Set(in_dim, 2, 1)
        # if args.transfer_mode ==1:
        self.in_dim = args.d_transfer
        # else:
        # 	self.in_dim  = in_dim
        self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)
        self.recons_type = args.recons_type

        self.batch_size = args.batch_size
        self.useAtt = args.useAtt
        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
        self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.num_nodes = -1
        self.device = args.device
        self.r_transfer_d = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, in_dim * 2))

        if args.task == "graph_regression":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
        elif args.task == "graph_classification":
            self.predict = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:
            print(f"checking mainmodel_finetuning task ...")

        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = GIN(self.in_dim, hidden_dim)
        elif encoder == "GCN":
            self.Encoder1 = GCN(self.in_dim, hidden_dim)
            self.Encoder2 = GCN(self.in_dim, hidden_dim)
        elif encoder == "GraphSAGE":
            self.Encoder1 = GraphSAGE(self.in_dim, hidden_dim)
            self.Encoder2 = GraphSAGE(self.in_dim, hidden_dim)
        elif encoder == "Transformer":  # in_dim, hidden_dim, num_layers, num_heads, device
            self.Encoder1 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
            self.Encoder2 = Transformer(self.in_dim, hidden_dim, num_layers, num_heads, args.device)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()

        # print(f"Loading pre-trained model .pt (Mainmodel_continue) ... ")
        self.model = torch.load(cp_filename, map_location=args.device)
        for p in self.model.parameters():
            p.requires_grad = True

        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1))
        self.reconstructX = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, in_dim))

    def compress(self, graph_features, device):
        p = self.compressor(graph_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        return gate_inputs, p

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        self.tau = 1
        num_nodes = z1.size(0)  # 32
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
                                                                                                                      i + 1) * batch_size].diag())))
        ret = torch.cat(losses)
        return ret.mean()

    def compression(self, nodes_list, device):
        epsilon = 0.0000001
        noisy_node_feature_all = torch.tensor(()).to(device)
        p_all = torch.tensor(()).to(device)
        KL_tensor_all = torch.tensor(()).to(device)

        z = len(nodes_list)
        graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
        for i in range(z):
            features = graph_feature_split[i]

            lambda_pos, p = self.compress(features, device)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_node_feature = features.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
            noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std
            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
                noisy_node_feature_mean) * noisy_node_feature_std

            noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

            p_all = torch.cat((p_all, p), 0)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
                ((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
            KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
        return noisy_node_feature_all, p_all, KL_tensor_all

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, current_epoch, edge_index,
                k_transition, device, batch_size=16):
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()
        self.device = device

        batch_x = self.transfer_d(batch_x)
        x_subs = self.transfer_d(x_subs)

        interaction_map, KL_tensor, noisy_node_feature, graph_features_readout = self.model.extract_features(nodes_list,
                                                                                                             batch_g,
                                                                                                             batch_x,
                                                                                                             flatten_batch_subgraphs,
                                                                                                             x_subs,
                                                                                                             device)

        interaction_map = self.MLP(interaction_map)

        # 6. KL upper bound
        KL_Loss = torch.mean(KL_tensor)

        # 7. Contrastive loss
        if self.readout == "sum":
            batch_g.ndata['h'] = noisy_node_feature
            noisy_node_feature_2 = dgl.sum_nodes(batch_g, 'h')  # [nb d]
        else:
            noisy_node_feature_2 = self.s2s(batch_g, noisy_node_feature)  # [nb , 2d]
        contrastive_loss = self.batched_semi_loss(noisy_node_feature_2, graph_features_readout, self.batch_size)

        # 8. Reconstruction loss
        if self.recons_type == 'adj':
            reconstruction_loss = self.loss_recon_adj(interaction_map, batch_g)
        elif self.recons_type == 'logM':
            reconstruction_loss = self.loss_recon(interaction_map, batch_logMs, nodes_list)
        else:
            reconstruction_loss = -1.0

        return None, KL_Loss, contrastive_loss, reconstruction_loss

    def loss_X(self, batch_x_org, interaction_map):
        # interaction_map_X = self.reconstructX(interaction_map)
        # loss = F.mse_loss(interaction_map, batch_x_org)
        row_num, col_num = interaction_map.size()
        loss = torch.sum((interaction_map - batch_x_org) ** 2) / (row_num)
        return loss

    def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, device):
        # 1. GNN encoder for graph batch
        graph_features = self.Encoder1(batch_g, batch_x)

        # 2. Subgraph extractions and Encoder
        subgraphs_features = self.Encoder2(flatten_batch_subgraphs, x_subs)

        # 3. Add normalization
        self.graph_features = graph_features  # F.normalize(graph_features, dim = 1)
        self.subgraphs_features = subgraphs_features  # F.normalize(subgraphs_features, dim = 1)

        # Pooling nb x 2d
        if self.readout == "sum":
            batch_g.ndata['h'] = self.graph_features
            graph_features_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
        else:
            graph_features_readout = self.s2s(batch_g, self.graph_features)  # [nb 2d]

        # 4. Compression p: preserve_rate
        noisy_node_feature, p, KL_tensor = self.compression(nodes_list, device)

        # 5. Core - subgraphs
        flatten_batch_subgraphs.ndata['h'] = self.subgraphs_features
        subgraphs_features_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'h')
        interaction_map = torch.cat((noisy_node_feature, subgraphs_features_readout), -1)

        # Attention-based interaction
        if self.useAtt:
            subgs_att = torch.tensor(()).to(device)
            if self.readout == "sum":
                batch_g.ndata['h'] = noisy_node_feature
                noisy_node_feature_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
            else:
                noisy_node_feature_readout = self.s2s(batch_g, noisy_node_feature)  # nb x 2d
                noisy_node_feature_readout = self.reduce_d(noisy_node_feature_readout)

            subgraphs_features_readout_split = torch.split(subgraphs_features_readout, tuple(nodes_list))
            z = len(subgraphs_features_readout_split)
            for i in range(z):
                noisy_node_feature_readout_cp = noisy_node_feature_readout[i].repeat(nodes_list[i], 1)
                interaction = torch.cat((noisy_node_feature_readout_cp, subgraphs_features_readout_split[i]),
                                        -1)  # x[n, d]     [n, d] -->  [n  2d]

                layer_atten = self.attn_layer(interaction)
                layer_atten = F.softmax(layer_atten, dim= 0)
                a = subgraphs_features_readout_split[i] * layer_atten  # [n , d]
                subgs_att = torch.cat((subgs_att, a), 0)
            interaction_map = torch.cat((noisy_node_feature, subgs_att), -1)
        return interaction_map, KL_tensor, noisy_node_feature, graph_features_readout

    #######################################################		end shared pre-training

    def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
        row_num, col_num = interaction_map.size()
        adj = org_graph.adj().to_dense()
        recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

        loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num)
        return loss

    def loss_recon(self, interaction_map, trans_logM, nodes_list):

        sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
        loss = 0
        z = len(nodes_list)
        for k in range(z):
            h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
            row_num, col_num = h.size()
            for i in range(self.k_transition):
                loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
                            row_num * col_num)
        loss = loss / (self.k_transition)
        return loss
class Transformer_class(nn.Module):
    def __init__(self, in_dim, out_dim, n_classes, hidden_dim, num_layers, num_heads, graph_name, cp_filename,
                 aug_check, sim_check, phi_check):
        super().__init__()

        print(f'Loading Transformer_class {cp_filename}')
        self.model = torch.load(cp_filename, map_location=args.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        self.model.aug_check = aug_check
        self.model.sim_check = sim_check
        self.model.phi_check = phi_check

        # if num_layers == 4:
        #     unfrezz_layers = ["layers." + str(num_layers), "layers." + str(num_layers - 1),
        #                       "layers." + str(num_layers - 2), "layers." + str(num_layers - 3)]
        # if num_layers == 3:
        #     unfrezz_layers = ["layers." + str(num_layers), "layers." + str(num_layers - 1),
        #                       "layers." + str(num_layers - 2)]
        # if num_layers == 2:
        #     unfrezz_layers = ["layers." + str(num_layers), "layers." + str(num_layers - 1)]

        # for name, para in self.model.named_parameters():
        #     # print(name)
        #     for layer in unfrezz_layers:
        #         if layer in name:
        #             para.requires_grad = True
        #         else:
        #             para.requires_grad = False

        for p in self.model.parameters():
            # model, '{}.pt'.format(ds_name))
            p.requires_grad = True

        # self.MLP = MLPReadout(out_dim, n_classes)
        self.MLP = MLP(out_dim, n_classes)

    # self.MLP2 = nn.Linear(out_dim, n_classes)
    # self.MLP2 = MLP(out_dim, n_classes)

    def forward(self, g, adj_org, sim, phi, B, k_transition, current_epoch, device, device_2):

        X = g.ndata['x'].to(device_2)
        edge_index = torch.stack([g.edges()[0], g.edges()[1]])

        h, _ = self.model.extract_features(g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device,
                                           device_2)

        h = self.MLP(h)
        h = F.softmax(h, dim=1)

        return h


class Transformer_Graph_class(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename):
        super().__init__()
        print(f'Loading Transformer_Graph_class {cp_filename}')
        self.model = torch.load(cp_filename, map_location=args.device)
        self.k_transition = k_transition
        for p in self.model.parameters():
            p.requires_grad = True

        # self.MLP_layer = MLPReadout(out_dim, 1)
        self.fc1 = torch.nn.Linear(out_dim, 1)

    # batch_graphs, batch_adj,batch_x,  1, edge_index, sim, phi,batch_B, 2, device, device)
    def forward(self, batch_g, batch_x, current_epoch, edge_index, k_transition, device):
        device = "cuda:0"
        k_transition = self.k_transition
        current_epoch = 1
        # extract_features(self, g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2):
        # h, _ = self.model.extract_features(batch_g, batch_adj_org, batch_x, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2)
        h, _ = self.model.extract_features(batch_g, batch_x, current_epoch, edge_index, k_transition, device)
        batch_g.ndata['h'] = h
        self.h = h

        hg = dgl.mean_nodes(batch_g, 'h')  # default readout is mean nodes
        # hg = self.MLP_layer(hg)
        # return F.softmax(hg, dim =1)
        sig = nn.Sigmoid()
        # print(f'hg: {hg}')

        return sig(self.fc1(hg))

    def loss(self, scores, targets):
        loss = nn.BCELoss()
        l = loss(scores.float(), targets.float())
        return l

# if torch.any(interaction_map.isnan()) :
# 	print(f"nan interaction_map ")