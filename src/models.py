import sklearn as sk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PERIOD = 10

class BTC_LSTM(torch.nn.Module):
    def __init__(self, dim_input=1, hidden_size=128, num_layers=2):
        super(BTC_LSTM,self).__init__()
        self.lstm = torch.nn.LSTM(input_size=dim_input, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_size, 1)
        
    def forward(self,x):
        y = torch.unsqueeze(x,-1)
        output,_ = self.lstm(y)
        output = self.fc(self.relu(output[:,-1]))        
        return output
        
class BTC_GRU(torch.nn.Module):
    def __init__(self, dim_input=1, hidden_size=128, num_layers=2):
        super(BTC_GRU,self).__init__()
        self.gru = torch.nn.LSTM(input_size=dim_input, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_size, 1)
        
    def forward(self,x):
        y = torch.unsqueeze(x,-1)
        output,_ = self.lstm(y)
        output = self.fc(self.relu(output[:,-1]))        
        return output
        
BTC_FC = torch.nn.Sequential(
    torch.nn.Linear(35,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256,512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256,1)
)
