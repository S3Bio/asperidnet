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


# In[12]:

class SpectralDataset:
    def __init__(self, filename, datafor='train'):
        data = pd.read_csv(filename)
        np.random.seed(0)
        data = data.sample(len(data)).reset_index(drop=True) #กำหนด random state
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

def func(trial):
    param = {
        'device' : 'cpu',
        'epoch_num' : 500,
        'input_size' : 1504,
        'classes' : 25,
        'batch_size' : 256,
        'lr' : trial.suggest_loguniform('lr', 1e-6, 1e-3),
        'optimizer_name' : 'Adam',}
    spectrum_train = SpectralDataset('df_newname.csv', datafor = 'train')
    spectrum_test = SpectralDataset('df_newname.csv', datafor = 'test')
    spectrum_val = SpectralDataset('df_newname.csv', datafor = 'validate')
    train_loader = DataLoader(spectrum_train, 
                          batch_size=param['batch_size'], 
                          shuffle=True)
    test_loader = DataLoader(spectrum_test,
                            batch_size=64,
                            shuffle=True)
    valid_loader = DataLoader(spectrum_val,
                            batch_size=64,
                            shuffle=True)
    

    def create_model():
        n_layers = trial.suggest_int('n_layers', 2, 16, log=True)
        layers = []
        for i in range(n_layers):
            output_size = trial.suggest_int('n_units_l{}'.format(i), 32, 2048, log=True)
            layers.append(nn.Linear(param['input_size'], output_size))
            layers.append(nn.ReLU())            
            param['input_size'] = output_size
        layers.append(nn.Linear(param['input_size'], param['classes']))
        layers.append(nn.LogSoftmax(dim=1))
        
        return nn.Sequential(*layers)
    
    model = create_model()
    optimizer = getattr(optim, param['optimizer_name'])(model.parameters(), lr=param['lr']) 
    criterion = nn.CrossEntropyLoss()

    for epoch in range(param['epoch_num']):
        model.train()
        training_loss = 0
        valid_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.float().to(param['device'])
            target = target.to(param['device'])
            target = target.type(torch.LongTensor)
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader)
         
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in valid_loader:
            inputs, target = batch
            inputs = inputs.float().to(param['device'])
            output = model(inputs)
            target = target.to(param['device'])
            target = target.type(torch.LongTensor)
            loss = criterion(output, target)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],target).view(-1) 
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        
    accuracy = num_correct / num_examples
    validation_loss = valid_loss/len(valid_loader)
    print('Training Loss= {:.5f}, Valid Loss= {:.5f}, Accuracy= {:.5f}'
              .format(training_loss, validation_loss, accuracy))
        
    return validation_loss


study = optuna.create_study(study_name='22_Oct', storage='sqlite:///irstudyOct.db', load_if_exists=True, direction="minimize")
study.optimize(func, n_trials=900)
#df_results = study.trials_dataframe(attrs=('number', 'value', 'params'))
#df_results.to_pickle('22_Oct.pkl')
#df_results.to_csv('22_Oct.csv')
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

