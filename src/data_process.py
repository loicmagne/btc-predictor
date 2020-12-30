import sklearn as sk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PERIOD = 10

#  Data set taken from http://www.cryptodatadownload.com/index.html
class DatasetBTC(torch.utils.data.Dataset):
    def __init__(self, csv_file, period=PERIOD):
        self.period = period
        
        # load dataset
        df = pd.read_csv(csv_file)
        dataset = df[['Date','Close']]
        dataset.columns = ['Date', 'Price']
        dataset = dataset.iloc[::-1]
        
        # scale datas
        self.scaler = StandardScaler()
        dataset[['Price']] = self.scaler.fit_transform(dataset[['Price']])
        
        self.length = len(dataset[[]])
        
        # compute dataset
        self.dataset = self.build_dataset(dataset)
        
    def get_sample(self, dataset, i):
        n = len(dataset)
        min_nb = 24*35
        if i < min_nb:
            return None
        if i >= n-self.period:
            return None
        datas = []
        for k in range(24):
            datas.append(dataset.iloc[i-k]['Price'])
        for k in range(7):
            datas.append(dataset.iloc[i-24*(k+1)]['Price'])
        for k in range(4):
            datas.append(dataset.iloc[i-(24*7)*(k+2)]['Price'])
        datas.reverse()
        return datas,dataset.iloc[i+self.period]['Price']

    def build_dataset(self, dataset):
        inputs = []
        labels = []
        for k in tqdm(range(self.length)):
            sample = self.get_sample(dataset,k)
            if sample is not None:
                inputs.append(sample[0])
                labels.append(sample[1])
        return np.array(inputs, dtype=np.double), np.array(labels, dtype=np.double).reshape(-1,1)
    
    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        return self.dataset[0][idx],self.dataset[1][idx]
        
def get_sample(prices, i, period=PERIOD):
    n = len(prices)
    min_nb = 24*35
    if i < min_nb and i >= n-period:
        return None
    datas = []
    for k in range(24):
        datas.append(prices[i-k])
    for k in range(7):
        datas.append(prices[i-24*(k+1)])
    for k in range(4):
        datas.append(prices[i-(24*7)*(k+2)])
    datas.reverse()
    return datas
