import torch
from torch.utils.data import DataLoader
from data import DataWrapper
from mlp import MLP
import numpy as np
import random
import torch.nn as nn
from pathlib import Path
import sys
from torch import optim
from data import load_walmart_data
from copy import deepcopy as dc
from config import get_config
import os

def fit(x_train, y_train, net, criterion, optimizer, bs, epochs, Y_max):

    train_data = DataLoader(DataWrapper(x_train, y_train), batch_size = bs, shuffle = True)

    for epoch in range(epochs):
            
        for batch_idx, (x, y) in enumerate(train_data):
            row_indices = torch.arange(x[:,:-1].size(0), device=x[:,:-1].device)
            pred = net(x[:,:-1])[row_indices, x[:,-1].to(torch.long)]*Y_max
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(net, test_store_data, y_dim, Y_max, h, b, gamma):
    y = 0
    costs = []
    for step in range(test_store_data.shape[0]):
        d = test_store_data[step,-1]
        feat = torch.tensor(test_store_data[step,:-1].astype(float)).float()
        actions = net(feat).detach().numpy()*Y_max
        x_index = int(((y + Y_max)/(2*Y_max)*y_dim))
        y = np.argmin(actions + (np.arange(y_dim) < x_index).astype(int)*1e5)/y_dim*2*Y_max-Y_max
        costs.append(gamma**step*(np.maximum(y - d, 0)*h + np.maximum(d - y, 0)*b))
        y = y - d

    return np.sum(costs)

def log(value, all_args):
    with open(all_args.log_dir, 'a+') as f:
        f.write(str(all_args.seed) + '_' + str(all_args.bs) + '_' + str(all_args.lr) + '_'  + str(all_args.hidden_size) + '_' + str(all_args.layer_num) + '_' + str(all_args.K) + '_' + str(all_args.y_dim) + '_' + str(all_args.gamma) + '_' + str(all_args.critical_ratio))
        f.write(' ' + str(value[-1]) + ' ' + str(1.96*np.std(value[:-1])/np.sqrt(len(value)-1)) + '\n')

def train(args):
    
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    all_train_store_data, all_test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)
    best_costs = []
    for train_store_data, test_store_data in zip(all_train_store_data, all_test_store_data):
        print(len(best_costs))
        feat_dim = int((train_store_data.shape[1]-1)/2)
        net = MLP(feat_dim, action_dim = all_args.y_dim, hidden_size = all_args.hidden_size, layer_N = all_args.layer_num)
        criterion =  nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr = all_args.lr, momentum = all_args.gamma_optimizer)
        test_best = np.inf
        for k in range(all_args.K):
            test_res = test(net, test_store_data, all_args.y_dim, all_args.Y_max, 1-all_args.critical_ratio, all_args.critical_ratio, all_args.gamma)
            test_best = min(test_best, test_res)
            old_net = dc(net)
            next_s = torch.tensor(train_store_data[:,feat_dim+1:].astype(float)).float()
            next_q = old_net(next_s).detach().numpy()*all_args.Y_max
            h_a = np.random.uniform(-all_args.Y_max, all_args.Y_max, train_store_data.shape[0])
            nw_cost = np.maximum(h_a - train_store_data[:,feat_dim], 0)*(1-all_args.critical_ratio) + np.maximum(train_store_data[:,feat_dim] - h_a, 0)*all_args.critical_ratio
            next_y_pos = ((np.maximum(h_a - train_store_data[:,feat_dim], -all_args.Y_max)+all_args.Y_max)/(2*all_args.Y_max)*all_args.y_dim).astype(int)
            next_opt_q = np.min(next_q + (np.arange(all_args.y_dim) < next_y_pos[:, np.newaxis]).astype(int)*1e5, axis=1)
            target_q = nw_cost + all_args.gamma*next_opt_q
            y_level = ((h_a + all_args.Y_max)/(2*all_args.Y_max)*all_args.y_dim).astype(int)[:,np.newaxis]
            feat = np.concatenate([train_store_data[:,:feat_dim], y_level], axis=1)
            fit(feat, target_q, net, criterion, optimizer, all_args.bs, all_args.epochs, all_args.Y_max)
        best_costs.append(test_best)
    best_costs.append(np.mean(best_costs))
    log(best_costs, all_args)

if __name__ == "__main__":
    train(sys.argv[1:])