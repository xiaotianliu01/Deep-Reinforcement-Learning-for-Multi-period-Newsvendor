import pandas as pd
import numpy as np
from copy import deepcopy as dc
import warnings 
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

binary_ecoder = {'A':[0, 0], 'B':[0, 1], 'C':[1, 0]}

def load_walmart_data(store_pth, train_pth, feature_pth, train_split_ratio = 0.8):

    store = pd.read_csv(store_pth)
    store['Size'] = store['Size']/store['Size'].max()
    store_num = store.shape[0]
    train = pd.read_csv(train_pth)
    train['Date'] = pd.to_datetime(train['Date'])
    feature = pd.read_csv(feature_pth)
    feature['Date'] = pd.to_datetime(feature['Date'])
    feature['IsHoliday'] = feature['IsHoliday'].astype(int)

    train_store_data = []
    test_store_data = []

    for sto in range(1, store_num+1):

        demand = train[(train['Store'] == sto) & (train['Dept'] == 1)][['Weekly_Sales']]
        feat = feature[(feature['Store'] == sto) & (feature['Date'] < pd.to_datetime("2012-11-2"))]
        feat.drop(columns=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
        feat['Week'] = feat.Date.dt.isocalendar().week
        feat['Year'] = feat.Date.dt.year
        feat_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'Year']
        for col in feat_columns:
            feat[col] = feat[col].fillna(feat[col].mean())
        sto_feat = dc(binary_ecoder[store[(store['Store'] == sto)].at[sto-1, 'Type']])
        sto_feat.append(store[(store['Store'] == sto)].at[sto-1, 'Size'])
        sto_feat = np.array([np.array(sto_feat) for _ in range(feat.shape[0])])
        sto_feat = pd.DataFrame(sto_feat, columns=['Type1', 'Type2', 'Size'])
        data = pd.concat([feat.reset_index(), sto_feat.reset_index(), demand.reset_index()], axis=1)

        for col in feat_columns:
            data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())
        data['Weekly_Sales'] = data['Weekly_Sales']/data['Weekly_Sales'].max()
        cols = feat_columns + ['IsHoliday'] + ['Type1', 'Type2', 'Size'] + ['Weekly_Sales']
        t_data_num = int(train_split_ratio*data.shape[0])
        train_d_t = np.array(data[cols])[:t_data_num,:]
        train_d_t = np.concatenate([train_d_t[:-1,:], train_d_t[1:,:-1]], axis=1)
        train_store_data.append(train_d_t)
        test_store_data.append(np.array(data[cols])[t_data_num:,:])

    return train_store_data, test_store_data

class DataWrapper(Dataset):

    def __init__(self, feature, target):
        super(DataWrapper, self).__init__()
        self.feature = torch.tensor(feature.astype(float), dtype=torch.float)
        self.target = torch.tensor(target.astype(float), dtype=torch.float)

    def __getitem__(self, index):
        item = self.feature[index]
        label = self.target[index]
        return item, label
    
    def __len__(self):
        return len(self.feature)