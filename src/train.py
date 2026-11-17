"""RelGNN model, temporal loader, training and inference, as in the official RelGNN examples
(https://github.com/snap-stanford/RelGNN, commit cffdb8b):
  * atomic routes, RelGNN_Model, temporal loader, BCE loss, best epoch by multilabel_auprc_macro: relgnn_task_node.py
  * 2 layers, 128 channels, sum aggregation, 128 neighbours halved per hop, directional subgraphs: utils.py
    (8 heads, one of the values used there)
  * Adam lr 0.005, 10 epochs: gnn_node.py; an epoch is capped at 500 steps instead of 2000 (about 75 min per model)
"""
import copy
import math
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from data import TARGETS
from relgnn.atomic_routes import get_atomic_routes
from relgnn.rel_graph import AttachTargetTransform, to_unix_time
from relgnn.relgnn_model import RelGNN_Model

ENTITY = "Users"
MODEL = dict(num_model_layers=2, channels=128, aggr="sum", num_heads=8)
LOADER = dict(num_neighbors=[128, 64], subgraph_type="directional", batch_size=512, temporal_strategy="uniform")
TRAIN = dict(lr=0.005, epochs=10, max_steps_per_epoch=500)


def make_loader(graph, table, shuffle, workers=8):
    """Temporal sampling: seeds at t - 1 s, so only rows with row time < t are used."""
    seed_time = torch.from_numpy(to_unix_time(table.time.astype("datetime64[ns]")) - 1)
    target = torch.from_numpy(table[TARGETS].to_numpy(np.float32))
    return NeighborLoader(graph, time_attr="time", input_nodes=(ENTITY, torch.from_numpy(table.user_id.to_numpy(np.int64))),
                          input_time=seed_time, transform=AttachTargetTransform(ENTITY, target), shuffle=shuffle,
                          num_workers=workers, persistent_workers=workers > 0, **LOADER)


def make_model(graph, col_stats, device):
    return RelGNN_Model(data=graph, col_stats_dict=col_stats, out_channels=len(TARGETS), norm="batch_norm",
                        atomic_routes=get_atomic_routes(graph.edge_types), **MODEL).to(device)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    return torch.cat([torch.sigmoid(model(batch.to(device), ENTITY)).cpu() for batch in loader]).numpy()


def fit(graph, col_stats, train_table, val_table, device, seed=0, log=print):
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # reproducible runs
    model = make_model(graph, col_stats, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader, val_loader = make_loader(graph, train_table, True), make_loader(graph, val_table, False)
    y_val = val_table[TARGETS].to_numpy()
    best_metric, best_state = -math.inf, None
    for epoch in range(1, TRAIN["epochs"] + 1):
        model.train()
        t0, loss_sum, count = time.time(), 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, ENTITY)
            loss = loss_fn(pred.float(), batch[ENTITY].y.float())
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * pred.size(0)
            count += pred.size(0)
            if step + 1 >= TRAIN["max_steps_per_epoch"]:
                break
        metric = average_precision_score(y_val, predict_proba(model, val_loader, device), average="macro")
        log(f"  epoch {epoch}: train loss {loss_sum / count:.4f}, validation AUPRC {metric:.4f} ({time.time() - t0:.0f}s)")
        if metric >= best_metric:
            best_metric, best_state = metric, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_metric
