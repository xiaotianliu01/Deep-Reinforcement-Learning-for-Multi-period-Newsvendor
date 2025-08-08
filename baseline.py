import numpy as np
from data import load_walmart_data
import sys
from config import get_config
import random
import torch.nn as nn
import scipy
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor, Ridge, Lasso
import xgboost as xgb
from mlp import MLP
import torch
from torch import optim
from torch.utils.data import DataLoader
from data import DataWrapper
from copy import deepcopy as dc

def log(value, all_args):
    print(value[-1])
    with open(all_args.log_dir, 'a+') as f:
        f.write(str(all_args.seed) + '_' + str(all_args.gamma) + '_' + str(all_args.critical_ratio) + ' ')
        f.write(str(value[-1]) + ' ' + str(1.96*np.std(value[:-1])/np.sqrt(len(value)-1)) + '\n')

def SAA(all_args):

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    train_store_data, test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)

    def get_emperical_quantile(h, b, demands):
        sorted_demand = np.sort(demands, axis=1)
        position = int(demands.shape[1]*b/(h+b))
        return sorted_demand[:,position]
    
    L = 100
    feat_dim = int((train_store_data[0].shape[1]-1)/2)
    train_data = np.concatenate([d[np.newaxis, :] for d in train_store_data], axis=0)[:,:,feat_dim]
    test_data = np.concatenate([d[np.newaxis, :] for d in test_store_data], axis=0)[:,:,-1]
    demands = np.concatenate([train_data[:,-L:], test_data], axis = 1)
    
    ys = np.zeros(test_data.shape[0])
    costs = []
    for step in range(L, demands.shape[1]):
        d = demands[:,step]
        his_d = demands[:,step-L:step]
        ys = np.max(np.concatenate([get_emperical_quantile(1-all_args.critical_ratio, all_args.critical_ratio, his_d)[:,np.newaxis], ys[:,np.newaxis]], axis=1), axis=1)
        costs.append(all_args.gamma**(step-L)*(np.maximum(ys - d, 0)*(1-all_args.critical_ratio) + np.maximum(d - ys, 0)*all_args.critical_ratio))
        ys = ys - d
    costs = list(np.sum(np.concatenate([c[:,np.newaxis] for c in costs], axis=1), axis = 1))
    costs.append(np.mean(costs))
    log(costs, all_args)

def PTO(all_args):

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    all_train_store_data, all_test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)
    costs = []
    L = 10
    for train_store_data, test_store_data in zip(all_train_store_data, all_test_store_data):

        feat_dim = int((train_store_data.shape[1]-1)/2)
        #train_x = np.concatenate([train_store_data[L:,:feat_dim]] + [train_store_data[l:-(L-l),feat_dim][:,np.newaxis] for l in range(L)], axis=1)
        #train_y = train_store_data[L:,feat_dim]
        train_x = train_store_data[:,:feat_dim]
        train_y = train_store_data[:,feat_dim]
        
        #rf = SGDRegressor(loss='epsilon_insensitive', epsilon=0)
        rf = xgb.XGBRegressor(max_depth=10, learning_rate=0.1)
        #rf = AdaBoostRegressor()
        rf.fit(train_x, train_y)
        
        #test_x = test_store_data[:,:feat_dim]
        past_d = np.array(list(train_store_data[-L:,feat_dim]) + list(test_store_data[:,-1]))
        #test_x = np.concatenate([test_x] + [past_d[l:-L+l][:,np.newaxis] for l in range(L)], axis=1)
        test_x = test_store_data[:,:-1]
        pred_ds = rf.predict(test_x)
        q = scipy.stats.norm.ppf(all_args.critical_ratio, loc=0, scale=1)
        y = 0
        c = []
        for step in range(test_store_data.shape[0]):
            pred_y = pred_ds[step] + q*np.std(past_d[step:step+L])
            d = test_store_data[step, -1]
            y = max(pred_y, y)
            c.append(all_args.gamma**step*(np.maximum(y - d, 0)*(1-all_args.critical_ratio) + np.maximum(d - y, 0)*all_args.critical_ratio))
            y = y - d
        costs.append(np.sum(c))
    costs.append(np.mean(costs))
    log(costs, all_args)

def SNN(all_args):

    class newsvendor_loss(nn.Module):
        def __init__(self, h=1.0, b=1.0):
            super().__init__()
            self.h = h
            self.b = b

        def forward(self, x, y):
            diff = x - y
            loss = torch.relu(diff) * self.h + torch.relu(-diff) * self.b
            return loss.mean()
    
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    all_train_store_data, all_test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)
    costs = []
    L = 15
    for train_store_data, test_store_data in zip(all_train_store_data, all_test_store_data):
        
        feat_dim = int((train_store_data.shape[1]-1)/2)
        net = MLP(feat_dim+L, action_dim = 1, hidden_size = all_args.hidden_size, layer_N = all_args.layer_num)
        criterion =  newsvendor_loss(1-all_args.critical_ratio, all_args.critical_ratio)
        optimizer = optim.SGD(net.parameters(), lr = all_args.lr, momentum = all_args.gamma_optimizer)
        x_train = np.concatenate([train_store_data[L:,:feat_dim]] + [train_store_data[l:-(L-l),feat_dim][:,np.newaxis] for l in range(L)], axis=1)
        y_train = train_store_data[L:,feat_dim]
        train_data = DataLoader(DataWrapper(x_train, y_train), batch_size = all_args.bs, shuffle = True)
        optimal_costs = np.inf
        for epoch in range(all_args.epochs):            
            for batch_idx, (x, y) in enumerate(train_data):
                pred = net(x)*0.5
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if(epoch%1 == 0):
                test_x = test_store_data[:,:feat_dim]
                past_d = np.array(list(train_store_data[-L:,feat_dim]) + list(test_store_data[:,-1]))
                test_x = np.concatenate([test_x] + [past_d[l:-L+l][:,np.newaxis] for l in range(L)], axis=1)
                test_x = torch.tensor(test_x.astype(float)).float()
                pred_ds = net(test_x).detach().numpy()*0.5
                y = 0
                c = []
                for step in range(test_store_data.shape[0]):
                    d = test_store_data[step, -1]
                    y = max(pred_ds[step][0], y)
                    c.append(all_args.gamma**step*(np.maximum(y - d, 0)*(1-all_args.critical_ratio) + np.maximum(d - y, 0)*all_args.critical_ratio))
                    y = y - d
                optimal_costs = min(np.sum(c), optimal_costs)
        costs.append(optimal_costs)
    costs.append(np.mean(costs))
    log(costs, all_args)

def E2E(all_args):

    class MQRNN(nn.Module):

        def __init__(self, feat_dim = 10, demand_length = 5, encoder_hidden_size=4, decoder_hidden_size=32, decoder_n_layers=6):
            
            super(MQRNN, self).__init__()
            self.encoder_hidden_size = encoder_hidden_size
            self.feat_dim = feat_dim
            self.decoder_n_layers = decoder_n_layers
            self.encoder = nn.LSTM(1, encoder_hidden_size, 1, bias=True, batch_first=True)
            self.fc1 = nn.Sequential(nn.Linear(encoder_hidden_size + demand_length, decoder_hidden_size), nn.Tanh())
            self.fc_h = nn.Sequential(nn.Linear(decoder_hidden_size, decoder_hidden_size), nn.Tanh())
            self.fc2 = nn.ModuleList([dc(self.fc_h) for i in range(decoder_n_layers)])
            self.fc3 = nn.Sequential(nn.Linear(decoder_hidden_size, 1), nn.Tanh())
        
        def forward(self, X):

            demand = X[:,self.feat_dim:]
            y = demand.unsqueeze(2)
            _, (h, c) = self.encoder(y)
            ht = h[-1, :, :]
            f = torch.concat([ht, X[:,self.feat_dim:]], axis=1)
            f = self.fc1(f)
            for i in range(self.decoder_n_layers):
                f = self.fc2[i](f)
            pred = self.fc3(f) + 1

            return pred
                
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    all_train_store_data, all_test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)
    costs = []
    L = 10
    for train_store_data, test_store_data in zip(all_train_store_data, all_test_store_data):
        
        feat_dim = int((train_store_data.shape[1]-1)/2)
        net = MQRNN(feat_dim, L, encoder_hidden_size=16, decoder_hidden_size=64, decoder_n_layers=6)
        criterion =  nn.L1Loss()
        optimizer = optim.SGD(net.parameters(), lr = all_args.lr, momentum = all_args.gamma_optimizer)
        x_train = np.concatenate([train_store_data[L:,:feat_dim]] + [train_store_data[l:-(L-l),feat_dim][:,np.newaxis] for l in range(L)], axis=1)
        y_train = train_store_data[L:,feat_dim]
        train_data = DataLoader(DataWrapper(x_train, y_train), batch_size = all_args.bs, shuffle = True)
        opt_costs = np.inf
        q = scipy.stats.norm.ppf(all_args.critical_ratio, loc=0, scale=1)

        for epoch in range(all_args.epochs): 

            for batch_idx, (x, y) in enumerate(train_data):
                pred = net(x)*0.5
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if(epoch%10 == 0):

                test_x = test_store_data[:,:feat_dim]
                past_d = np.array(list(train_store_data[-L:,feat_dim]) + list(test_store_data[:,-1]))
                test_x = np.concatenate([test_x] + [past_d[l:-L+l][:,np.newaxis] for l in range(L)], axis=1)
                test_x = torch.tensor(test_x.astype(float)).float()
                pred_ds = net(test_x).detach().numpy()*0.5
                y = 0
                c = []
                for step in range(test_store_data.shape[0]):
                    pred_y = pred_ds[step][0] + q*np.std(past_d[step:step+L])
                    d = test_store_data[step, -1]
                    y = max(pred_y, y)
                    c.append(all_args.gamma**step*(np.maximum(y - d, 0)*(1-all_args.critical_ratio) + np.maximum(d - y, 0)*all_args.critical_ratio))
                    y = y - d
                opt_costs = min(np.sum(c), opt_costs)
        costs.append(opt_costs)
    costs.append(np.mean(costs))
    log(costs, all_args)

def KO(all_args):
    
    def gaussian_kernel(x):
        return 1/np.sqrt(2*np.pi)*np.exp((-np.sum(np.square(x), axis=1)/2).astype(float))
    
    def get_opt_y(his_data, current_feat, h, b):
        his_feat = his_data[:,:-1]
        his_d = his_data[:,-1]
        kappas = gaussian_kernel(his_feat-current_feat)
        all_his_data = []
        for i in range(kappas.shape[0]):
            all_his_data.append([kappas[i], his_d[i]])
        all_his_data.sort(key=lambda x: x[1])
        t = 0
        for n in range(kappas.shape[0]):
            t += all_his_data[n][0]
            if(t/np.sum(kappas)>b/(h+b)):
               break
        return all_his_data[n][1]

    all_train_store_data, all_test_store_data = load_walmart_data('./data/stores.csv', './data/train.csv', './data/features.csv', 0.8)
    L = 50
    costs = []
    for train_store_data, test_store_data in zip(all_train_store_data, all_test_store_data):
        feat_dim = int((train_store_data.shape[1]-1)/2)
        features = np.concatenate([train_store_data[-L:,:feat_dim+1], test_store_data], axis = 0)
        y = 0
        c = []
        for step in range(L, features.shape[0]):
            his_data = features[step-L:step,:]
            current_feat = features[step,:-1]
            d = features[step,-1]
            y = max(get_opt_y(his_data, current_feat, 1-all_args.critical_ratio, all_args.critical_ratio), y)
            c.append(all_args.gamma**(step-L)*(np.maximum(y - d, 0)*(1-all_args.critical_ratio) + np.maximum(d - y, 0)*all_args.critical_ratio))
            y = y - d
        costs.append(np.sum(c))
    costs.append(np.mean(costs))
    log(costs, all_args)
    print('a')

if __name__ == "__main__":

    parser = get_config()
    all_args = parser.parse_known_args(sys.argv[1:])[0]
    if('saa' in all_args.exp_name):
        SAA(all_args)
    if('pto' in all_args.exp_name):
        PTO(all_args)
    if('snn' in all_args.exp_name):
        SNN(all_args)
    if('e2e' in all_args.exp_name):
        E2E(all_args)
    if('ko' in all_args.exp_name):
        KO(all_args)