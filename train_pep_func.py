import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU, MAE


def cosinSim(x_hat):
    x_norm = torch.norm(x_hat, p=2, dim=1)
    nume = torch.mm(x_hat, x_hat.t())
    deno = torch.ger(x_norm, x_norm)
    cosine_similarity = nume / deno
    return cosine_similarity


def train_epoch(model, args, optimizer, device, data_loader, epoch, k_transition, batch_size=16):
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


from ogb.graphproppred import Evaluator


def process_diff(batch_adj, batch_size):
    list_batch = []
    max_size = 0
    # print(f"batch_adj[i]: {batch_adj[0]}")
    for i in range(batch_size):
        size = batch_adj[i].size(dim=1)
        if size > max_size:
            max_size = size

    p2d = (0, 2, 0, 2)  # pad last dim by 1 on each side
    for i in range(batch_size):
        diff = max_size - batch_adj[i].size(dim=1)
        if diff != max_size:
            p2d = (0, diff, 0, diff)  # pad last dim by 1 on each side
            batch_adj[i] = F.pad(batch_adj[i], p2d, "constant", 0)
            # print(f"batch_adj[i]: {batch_adj[i].size()}")
            list_batch.append(batch_adj[i])
    return torch.stack(batch_adj, dim=0)


# raise SystemExit()
import dgl
from itertools import chain
from MetricWrapper import MetricWrapper


def train_epoch_domainadaptation(model, args, optimizer, device, data_loader, epoch, k_transition, batch_size=16):
    model.train()

    epoch_loss = 0;
    epoch_reconstruction_loss = 0
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
        reconstruction_loss = model.forward(batch_graphs, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, 1,
                                            edge_index, 2, device, batch_size)
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_reconstruction_loss += reconstruction_loss

    epoch_loss /= (count + 1)
    epoch_reconstruction_loss /= (count + 1)
    return epoch_loss, epoch_reconstruction_loss


from metrics import eval_ap


def train_epoch_graph_classification(args, model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    wrapped_loss_fun = MetricWrapper(metric=model.loss, target_nan_mask="ignore-flatten")
    evaluator = Evaluator(name="Peptides-func")
    epoch_loss = 0
    epoch_train_ap = 0
    targets = torch.tensor([]).to(device)
    scores = torch.tensor([]).to(device)
    epoch_loss = 0

    # for iter, (batch_graphs, batch_targets, _, batch_B, batch_adj, batch_sim, batch_phi ) in enumerate(data_loader):
    for iter, (batch_graphs, batch_targets, batch_subgraphs, _) in enumerate(data_loader):
        # count += 1
        # if count % 300 ==0:
        #     print(f'Processing batches: {count}')
        batch_targets = batch_targets.to(device)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].float().to(device)  # num x feat
        # sim = batch_graphs.edata['sim'].float().to(device)  # num x feat
        edge_index = batch_graphs.edges()

        optimizer.zero_grad()

        flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))

        flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)  # batch_subgraphs.to(device)
        x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)  # num x feat
        batch_x = F.normalize(batch_x)
        x_subs = F.normalize(x_subs)
        batch_scores, _, _, _ = model.forward(batch_graphs, batch_x, flatten_batch_subgraphs, x_subs, 1, edge_index, 2,
                                              device, batch_size)
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))


        loss = wrapped_loss_fun(batch_scores, batch_targets)
        loss = loss / 2

        loss.backward()

        # weights update
        if ((iter + 1) % 2 == 0) or (iter + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.detach().item()
        targets = torch.cat((targets, batch_targets.detach()), 0)
        scores = torch.cat((scores, batch_scores.detach()), 0)

        epoch_train_ap += model.BCEWithLogitsLoss(batch_scores, batch_targets)

    input_dict = {"y_true": targets, "y_pred": scores}
    #epoch_train_ap = evaluator.eval(input_dict)['ap']
    epoch_train_ap /= (iter + 1)
    epoch_loss /= (iter + 1)
    return epoch_loss, epoch_train_ap.detach().cpu(), optimizer


def evaluate_network(args, model, optimizer, device, data_loader, epoch,
                     batch_size):  # (model, device, data_loader, epoch):
    model.eval()
    evaluator = Evaluator(name="Peptides-func")

    epoch_test_loss = 0

    targets = torch.tensor([])
    scores = torch.tensor([])
    epoch_test_ap = 0
    wrapped_loss_fun = MetricWrapper(metric=model.loss, target_nan_mask="ignore-flatten")

    with torch.no_grad():
        # for iter, (batch_graphs, batch_targets, _) in enumerate(data_loader):
        for iter, (batch_graphs, batch_targets, batch_subgraphs, _) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['x'].float().to(device)  # num x feat
            edge_index = batch_graphs.edges()

            batch_targets = batch_targets.to(device)
            # batch_subgraphs = batch_subgraphs.to(device)

            optimizer.zero_grad()
            flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))

            flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)  # batch_subgraphs.to(device)
            x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)  # num x feat

            batch_x = F.normalize(batch_x)
            x_subs = F.normalize(x_subs)
            batch_scores, _, _, _ = model.forward(batch_graphs, batch_x, flatten_batch_subgraphs, x_subs, 1, edge_index,
                                                  2, device)

            loss = wrapped_loss_fun(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            targets = torch.cat((targets, batch_targets.detach().cpu()), 0)
            scores = torch.cat((scores, batch_scores.detach().cpu()), 0)

            epoch_test_ap += model.BCEWithLogitsLoss(batch_scores, batch_targets)

    #input_dict = {"y_true": targets, "y_pred": scores}
    #epoch_test_ap = evaluator.eval(input_dict)['ap']
    epoch_test_ap /= (iter + 1)
    epoch_test_loss /= (iter + 1)

    return epoch_test_loss, epoch_test_ap.detach().cpu()
