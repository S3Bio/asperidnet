#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class LightningNeuralNetIr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(1504, hidden1_size)
        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_3 = nn.Linear(hidden2_size, 26)
        
    def forward(self, x):
        x = x.view(-1, 1504)
        x = F.ReLU(self.layer_1(x))
        x = F.ReLU(self.layer_2(x))
        x = nn.LogSoftmax(self.layer_3(x), dim=1)
        return x
    
    def set_optimizers(self):
        param = {
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'epoch_num' : 100,
        'input_size' : 1504,
        'classes' : 26,
        'batch_size' : 64,
        'lr' : 1e-3,
        'optimizer_name' : 'Adam',
        }
        
        optimizer = getattr(optim, param['optimizer_name'])(self.parameters(), lr=param['lr'])
        return optimizer
    
    def optimizer_step(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()
    
    def set_loss(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def training_step(self, train_batch):
        inputs, target = train_batch
        inputs = inputs.float().to(param['device'])
        target = target.to(param['device'])
        output = self.forward(x)
        loss = self.set_loss(output, target)
        return loss
    
    def val_step(self, val_batch):
        inputs, target = val_batch
        inputs = inputs.float().to(param['device'])
        target = target.to(param['device'])
        output = self.forward(x)
        loss = self.set_loss(output, target)

class LightSpectralDataset(pl.LightningModule):
    def __init__(self, filename, datafor='train'):
        data = pd.read_csv(filename)
        np.random.seed(0)
        data = data.sample(len(data)).reset_index(drop=True)
        cols =[x for x in data.columns if x not in ['target']]
        rowused = []
        for i in range(len(data)):
            if i % 10 == 0:
                rowused.append('test')
                
            elif i % 10 == 1:
                rowused.append('validate')
                
            else:
                rowused.append('train')
                            
        data['rowused'] = rowused
        if datafor == 'train':
            selected_data = data.loc[data['rowused'] == 'train', :]
        elif datafor == 'validate':
            selected_data = data.loc[data['rowused'] == 'validate', :]
        elif datafor == 'test':
            selected_data = data.loc[data['rowused'] == 'test', :]
        elif datafor == 'all':
            selected_data = data
        else:
            raise("datafor supports only test train and validate")
        
    self.X = selected_data[cols]
    le = LabelEncoder()
    self.y = le.fit_transform(selected_data.loc[:, 'target'].values.ravel())
    self.y = self.y.reshape(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (self.X.iloc[index, :].to_numpy(), self.y[index])

    def train_dataloader(self):
        self.spectrum_train = LightSpectralDataset('data2.csv', datafor = 'train')
        return DataLoader(self.spectrum_train, batch_size=64, shuffle=True
    
    def test_dataloader(self):
        self.spectrum_test = LightSpectralDataset('data2.csv', datafor = 'test')
        return DataLoader(self.spectrum_test, batch_size=64, shuffle=True)
    
    def val_dataloader(self):
        self.spectrum_val = LightSpectralDataset('data2.csv', datafor = 'validate')
        return DataLoader(self.spectrum_val, batch_size=64, shuffle=True)

data_module = LightSpectralDataset()

#train

model = LightningNeuralNetIr()
                          
trainer = pl.Trainer()
trainer.fit(model, data_module)
    
    
    
        

