import sklearn as sk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models import *
from data_process import *

n_epochs = 10
learning_rate = 1e-2
model = BTC_LSTM()
model = model.double()
loss_fn = torch.nn.MSELoss()
criterion = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = DatasetBTC('BTCUSD_1hr.csv')
test_size = int(len(dataset)/10)
train_size = len(dataset) - test_size
print("Dataset size : {}, Trainset size :{}, Testset size:{}".format(len(dataset),train_size,test_size))
train_set, test_set = torch.utils.data.random_split(dataset,[train_size,test_size])
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

train_loss_history = []
test_loss_history = []
eps = 1e-2
lbda = 1e-3
for k in range(n_epochs):
    model.train()
    for inputs, labels in tqdm(train_dataloader):
        pred = model(inputs)
        loss = loss_fn(pred,labels) + lbda*torch.sum(torch.abs(inputs[:,-1]-labels)/(eps+torch.abs(pred-inputs[:,-1])))
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss_history.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        loss = 0
        for inputs, labels in test_dataloader:
            pred = model(inputs)
            l = criterion(pred,labels)
            loss += l
        test_loss_history.append(loss/len(test_set))
        
plt.plot(train_loss_history)
plt.show()
plt.plot(test_loss_history)
plt.show()

btc_prices = pd.read_csv('BTCUSD_1hr.csv')
btc_prices = btc_prices[['Date','Close']]
btc_prices.columns = ['Date', 'Price']
btc_prices = btc_prices.iloc[::-1]
prices = btc_prices['Price'].to_list()
dates = btc_prices['Date'].to_list()

predicted_prices = prices[:841]
model.eval()
for k in tqdm(range(840,len(prices)-24)):
    sample = torch.tensor(dataset.get_sample(btc_prices,k)[0])
    sample = dataset.scaler.transform(sample.unsqueeze(1)).squeeze()
    with torch.no_grad():
        pred = model(torch.tensor(sample).reshape(1,-1))
        pred = dataset.scaler.inverse_transform(pred).item()
        predicted_prices.append(pred)
        
n = len(predicted_prices)
for k in tqdm(range(20000)):
    sample = torch.tensor(get_sample(predicted_prices,k+n-24))  
    sample = dataset.scaler.transform(sample.unsqueeze(1)).squeeze()
    with torch.no_grad():
        pred = model(torch.tensor(sample).reshape(1,-1)) #+ np.random.randn()/100
        pred = dataset.scaler.inverse_transform(pred).item()
        predicted_prices.append(pred)
        
plt.figure(figsize=(20,10))
plt.plot(prices[45000:46000])
plt.plot(predicted_prices[45000:46000])
plt.show()
plt.figure(figsize=(20,10))
plt.plot(prices[45380:45400])
plt.plot(predicted_prices[45380:45400])
plt.show()
plt.figure(figsize=(20,10))
plt.plot(prices)
plt.plot(predicted_prices)
plt.show()
