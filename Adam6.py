#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import optuna
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from optuna.study import StudyDirection
from optuna._study_summary import StudySummary

data2 = pd.read_csv('data2.csv')
class SpectralDataset:
    def __init__(self, filename, datafor='train'):
        data = pd.read_csv(filename)
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
        self.y = le.fit_transform(selected_data.loc[:, 'target'].values.reshape(-1, 1))
        self.y = self.y.reshape(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (self.X.iloc[index, :].to_numpy(), self.y[index])


# # Train model

# In[6]:


#imports and data loaders

def func(trial):
    param = {
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'epoch_num' : 100,
        'input_size' : 1504,
        'hidden_sizes' : trial.suggest_int('hidden_sizes', 1, 50),
        'output_size' : 26,
        'batch_size' : 128,
        'lr' : trial.suggest_loguniform('lr', 1e-5, 1),
        'optimizer_name' : trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
    }
    spectrum_train = SpectralDataset('data2.csv', datafor = 'train')
    spectrum_test = SpectralDataset('data2.csv', datafor = 'test')
    #spectrum_val = SpectralDataset('data2.csv', datafor = 'validate')
    train_loader = DataLoader(spectrum_train, 
                          batch_size=param['batch_size'], 
                          shuffle=True)
    test_loader = DataLoader(spectrum_test,
                            batch_size=param['batch_size'],
                            shuffle=True)
    # valid_loader = DataLoader(spectrum_val,
    #                         batch_size=param['batch_size'],
    #                         shuffle=True)
    
#         print('Epoch: {}, accuracy = {:.2f}'.format(epoch, num_correct / num_examples))

    def create_model():
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layers = []
        layers.append(nn.Linear(param['input_size'], param['hidden_sizes']))
        layers.append(nn.ReLU())
        for i in range(n_layers):
            layers.append(nn.Linear(param['hidden_sizes'], param['hidden_sizes']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(param['hidden_sizes'], param['output_size']))
        return nn.Sequential(*layers)
    
    model = create_model()
    optimizer = getattr(optim, param['optimizer_name'])(model.parameters(), lr=param['lr']) 
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    num_correct = 0
    num_examples = 0
    for epoch in range(param['epoch_num']):        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.float().to(param['device'])
            target = target.to(param['device'])
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
         
        model.eval()
        for batch in test_loader:
            inputs, target = batch
            inputs = inputs.float().to(param['device'])
            output = model(inputs)
            target = target.to(param['device'])
            loss = criterion(output, target)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],target).view(-1) 
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        
    accuracy = num_correct / num_examples
        
    return accuracy    


study = optuna.create_study(study_name='Adam_6', storage='sqlite:///irstudy7.db', load_if_exists=True, direction="maximize")
study.optimize(func, n_trials=1000)
df_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
df_results.to_pickle('Adam_6.pkl')
df_results.to_csv('Adam_6.csv')
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))