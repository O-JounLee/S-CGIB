import argparse
import copy
import logging
import math
from molecules import MoleculeDataset
import dgl
from itertools import chain
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU, MAE
import time
from pathlib import Path
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os
import random
import dgl
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')
from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
    load_film, load_airports, load_amazon, load_coauthor, load_WikiCS, load_crocodile, load_Cora_ML

from util import get_B_sim_phi, getM_logM, load_dgl, get_A_D, load_dgl_fromPyG

from models import Transformer, Mainmodel, Mainmodel_finetuning

# from script_classification import run_node_classification, run_epoch_node_classification, update_evaluation_value, run_node_clustering


from script_classification import run_node_classification, run_node_clustering, update_evaluation_value

np.seterr(divide='ignore')


def collate(self, samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)

    return batched_graph, labels


from models import Mainmodel, Mainmodel_continue
from torch.utils.data import DataLoader


def run_pretraining(model, pre_train_loader1, optimizer, batch_size, device):
    best_epoch = 0
    best_model = model
    best_loss = 100000000
    for epoch in range(1, args.pt_epoches):
        epoch_train_loss, KL_Loss, contrastive_loss, reconstruction_loss = train_epoch_pre_training(model, args,
                                                                                                    optimizer, device,
                                                                                                    pre_train_loader1,
                                                                                                    epoch, 1,
                                                                                                    batch_size)
        if best_loss >= epoch_train_loss:
            best_model = model
            best_epoch = epoch
            best_loss = epoch_train_loss
        if epoch - best_epoch > 50:
            break
        if epoch % 1 == 0:
            msg = "Epoch:%d	|Best_epoch:%d	|Train_loss:%0.4f" % (epoch, best_epoch, epoch_train_loss)
            print(msg)
    return best_model, best_epoch


def run(i, dataset_full1, feature1, dataset_full2, feature2, dataset_full3, feature3):
    model = Mainmodel(args, feature1, hidden_dim=args.dims, num_layers=args.num_layers,
                      num_heads=args.num_heads, k_transition=args.k_transition, encoder=args.encoder).to(device)
    best_model = model
    best_loss = 100000000
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    data_all1 = dataset_full1.data_all
    data_all2 = dataset_full2.data_all
    data_all3 = dataset_full3.data_all

    batch_size = args.batch_size

    pre_train_loader1 = DataLoader(data_all1, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_full1.collate)  # Pre-training model
    pre_train_loader2 = DataLoader(data_all2, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_full2.collate)  # Pre-training model
    pre_train_loader3 = DataLoader(data_all3, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_full3.collate)  # Pre-training model

    if args.pretrained_mode == 1:
        # if not os.path.exists(file_name_cpt):
        file_name_cpt = args.output_path + f'pre_training_{args.dataset_list[0]}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'

        # 1
        if not os.path.exists(file_name_cpt):
            torch.save(best_model, file_name_cpt)

            model = Mainmodel_continue(args, feature1, hidden_dim=args.dims, num_layers=args.num_layers,
                                       num_heads=args.num_heads, k_transition=args.k_transition, num_classes=1,
                                       cp_filename=file_name_cpt, encoder=args.encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
            best_model, _ = run_pretraining(model, pre_train_loader1, optimizer, batch_size, device)
            torch.save(best_model, file_name_cpt)
            time.sleep(0.1)
        print(f"Finished pre-trained model step 1...")

        # 2
        file_check = args.output_path + f'pre_training_{args.dataset_list[0]}_{args.dataset_list[1]}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
        if not os.path.exists(file_check):
            model = Mainmodel_continue(args, feature2, hidden_dim=args.dims, num_layers=args.num_layers,
                                       num_heads=args.num_heads, k_transition=args.k_transition, num_classes=1,
                                       cp_filename=file_name_cpt, encoder=args.encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
            best_model, _ = run_pretraining(model, pre_train_loader2, optimizer, batch_size, device)
            file_name_cpt = args.output_path + f'pre_training_{args.dataset_list[0]}_{args.dataset_list[1]}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
            torch.save(best_model, file_name_cpt)
        print(f"Finished pre-trained model step 2...")

        # 3
        if len(args.dataset_list) == 3:
            file_check = args.output_path + f'pre_training_{args.dataset_list[0]}_{args.dataset_list[1]}_{args.dataset_list[2]}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
            if not os.path.exists(file_check):
                model = Mainmodel_continue(args, feature3, hidden_dim=args.dims, num_layers=args.num_layers,
                                           num_heads=args.num_heads, k_transition=args.k_transition, num_classes=1,
                                           cp_filename=file_name_cpt, encoder=args.encoder).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
                best_model, _ = run_pretraining(model, pre_train_loader3, optimizer, batch_size, device)
                file_name_cpt = args.output_path + f'pre_training_{args.dataset_list[0]}_{args.dataset_list[1]}_{args.dataset_list[2]}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
                torch.save(best_model, file_name_cpt)
            print(f"Finished pre-trained model step 3...")

    print(f"\nFinished pretraining models on {str(args.dataset_list)} ...")

    return 0


##################################################################################################################################
##################################################################################################################################
import collections
from collections import defaultdict
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset, ZINC, QM9
from dgl.data.utils import save_graphs, load_graphs


#    dataset = ["FreeSolv", "ESOL", "SIDER", "ToxCast", "ClinTox", "Tox21", "BACE", "BBBP", "PROTEINS", "ENZYMES",
# "NCI1", "NCI109", "ZINC", "ogbg-molpcba", "ogbg-molhiv"]
def get_num_features(ds):
    dim = args.feature_list
    if len(args.dataset_list) != len(dim):
        print(f"len features erro ")
        raise SystemExit()
    f = -1
    for i in range(len(args.dataset_list)):
        if args.dataset_list[i] == ds:
            f = dim[i]
    return f


def load_graphdataset(dataset_name):
    args.dataset = dataset_name
    args.num_features = get_num_features(dataset_name)
    graph_lists = []
    graph_labels = []

    print(f'Checking dataset {args.dataset}')
    graph_lists, graph_labels = load_graphs("pts/" + dataset_name + "_k_transition_" + str(args.k_transition) + ".bin")
    print(f"# 4 Loading subgraphs ...")
    subgraph_lists = torch.load("pts/" + dataset_name + "_subgraphs_khop_" + str(args.k_transition) + ".pt")  #
    subgraph_lists = subgraph_lists['set_subgraphs']
    print(f"len(subgraph_lists): {len(subgraph_lists)} | len(graph_lists): {len(graph_lists)}")

    dic = torch.load("pts/" + dataset_name + "_M_khop_" + str(args.k_transition) + ".pt")
    trans_logMs = dic['trans_logMs']
    samples_all = []
    num_features = args.num_features
    print(f'num_features: {num_features} ')
    samples_all = []
    checking_label = []
    num_node_list = []
    for i in range(len(graph_lists)):
        current_graph = graph_lists[i]

        current_label = graph_labels['glabel'][i]
        checking_label.append(current_label)
        num_node_list.append(current_graph.num_nodes())
        current_subgraphs = subgraph_lists[i]
        current_trans_logM = trans_logMs[i]

        pair = (current_graph, current_label, current_subgraphs, current_trans_logM)
        samples_all.append(pair)
    random.shuffle(samples_all)
    dataset_full = LoadData(samples_all, 'pre_training')

    return dataset_full, num_features


def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = args.dataset + "-" + timestr + ".log"
    Path("./exp_logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
    logging.info("Starting on device: %s", device)
    logging.info("Config: %s ", args)

    args.output_path = "outputs/"
    args.dataset_list = ['PCQM4Mv2', 'QM9', 'mol-PCBA']
    args.feature_list = [9, 11, 9]

    dataset_full1, feature1 = load_graphdataset(args.dataset_list[0])
    dataset_full2, feature2 = load_graphdataset(args.dataset_list[1])
    dataset_full3, feature3 = load_graphdataset(args.dataset_list[2])
    # dataset_full4, feature4 = load_graphdataset(args.dataset_list[3])

    runs_acc = []
    for i in tqdm(range(args.run_times)):
        acc = run(i, dataset_full1, feature1, dataset_full2, feature2, dataset_full3, feature3)
        runs_acc.append(acc)


def LoadData(samples_all, DATASET_NAME):
    return MoleculeDataset(samples_all, DATASET_NAME)


def generate_graphs(dataset, k_hop):
    graph_ds = GraphClassificationDataset()

    graph_labels = []

    set_subgraphs = []

    trans_logMs = []
    miss = 0
    checking = []

    for i in range(len(dataset)):
        # if i >= 200:
        # 	print(f" testing small dataset")
        # 	break
        if i % 10 == 0:
            print(f'Processing graph_th: {i}')
            time.sleep(0.1)
        data = dataset[i]

        path = "pts/" + args.dataset + "_kstep_" + str(args.k_transition) + ".pt"
        try:
            g = load_dgl_fromPyG(data)
            if not os.path.exists(path):
                M, logM = load_bias(g)
                trans_logM = torch.from_numpy(np.array(logM)).float()
            graph_ds.graph_lists.append(g)

            trans_logMs.append(trans_logM)
            graph_labels.append(data.y)

            checking.append(data.y[0])
            ####adding set subgraphs
            node_ids = g.nodes()

            all_subgraphs = [dgl.khop_in_subgraph(g, individual_node, k=args.k_transition)[0] for individual_node in
                             node_ids]

            set_subgraphs.append(all_subgraphs)

        except:
            miss += 1
            print(f'Missing loading dgl graph: {i}')
    graph_labels = torch.stack(graph_labels)
    graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}
    print(f"total DGL missing: {miss}")

    torch.save({"set_subgraphs": set_subgraphs}, "pts/" + args.dataset + "_subgraphs_khop_" + str(k_hop) + ".pt")
    # if os.path.exists("pts/"+ args.dataset + "_10p_M.pt") == False:
    torch.save({"trans_logMs": trans_logMs}, "pts/" + args.dataset + "_M_khop_" + str(k_hop) + ".pt")

    return graph_ds


def train_epoch_pre_training(model, args, optimizer, device, data_loader, epoch, k_transition, batch_size=16):
    model.train()

    epoch_loss = 0;
    epoch_KL_Loss = 0;
    epoch_contrastive_loss = 0;
    epoch_reconstruction_loss = 0
    nb_data = 0
    gpu_mem = 0
    count = 0
    for iter, (batch_graphs, _, batch_subgraphs, batch_logMs) in enumerate(data_loader):

        count = iter
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].float().to(device)
        edge_index = batch_graphs.edges()

        optimizer.zero_grad()
        flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
        flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)
        x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)

        batch_x = F.normalize(batch_x)

        x_subs = F.normalize(x_subs)
        _, KL_Loss, contrastive_loss, reconstruction_loss = model.forward(batch_graphs, batch_x,
                                                                          flatten_batch_subgraphs, batch_logMs, x_subs,
                                                                          1, edge_index, 2, device, batch_size)
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        loss = KL_Loss + reconstruction_loss + contrastive_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_KL_Loss += KL_Loss;
        epoch_contrastive_loss += contrastive_loss;
        epoch_reconstruction_loss += reconstruction_loss

    epoch_loss /= (count + 1)
    epoch_KL_Loss /= (count + 1);
    epoch_contrastive_loss /= (count + 1);
    epoch_reconstruction_loss /= (count + 1)
    return epoch_loss, epoch_KL_Loss, epoch_contrastive_loss, epoch_reconstruction_loss


class GraphClassificationDataset:
    def __init__(self):
        self.graph_lists = []  # A list of DGLGraph objects
        self.graph_labels = []
        self.subgraphs = []

    def add(self, g):
        self.graph_lists.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        # Get the i^th sample and label
        # return self.graphs[i], self.labels[i], self.trans_logM[i], self.B[i], self.adj[i], self.sim[i], self.phi[i]
        return self.graphs[i], self.labels[i], self.subgraphs[i]


def load_bias(g):
    M, logM = getM_logM(g, kstep=args.k_transition)
    return M, logM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments")

    #
    parser.add_argument("--dataset", default="pre-train", help="Dataset")
    parser.add_argument("--model", default="Mainmodel", help="GNN Model")

    parser.add_argument("--run_times", type=int, default=1)

    parser.add_argument("--drop", type=float, default=0.1, help="dropout")
    parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

    # adding args
    parser.add_argument("--device", default="cuda:0", help="GPU ids")

    # transfer learning
    parser.add_argument("--pretrained_mode", type=int, default=1)
    parser.add_argument("--domain_adapt", type=int, default=0)

    parser.add_argument("--d_transfer", type=int, default=32)
    parser.add_argument("--layer_relax", type=int, default=0)
    parser.add_argument("--readout_f", default="sum")  # mean set2set sum
    # transfer learning

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--testmode", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--pt_epoches", type=int, default=100)
    parser.add_argument("--ft_epoches", type=int, default=100)
    parser.add_argument("--useAtt", type=int, default=1)
    parser.add_argument("--dims", type=int, default=64, help="hidden dims")
    parser.add_argument("--task", default="graph_classification")
    parser.add_argument("--encoder", default="GIN")
    parser.add_argument("--recons_type", default="adj")
    parser.add_argument("--k_transition", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--output_path", default="outputs/", help="outputs model")

    parser.add_argument("--pre_training", default="1", help="pre_training or not")
    parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
    parser.add_argument("--file_name", default="outputs_excels.xlsx", help="file_name dataset")

    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    main()




