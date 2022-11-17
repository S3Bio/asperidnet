#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import optuna
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SpectralDataset:
    def __init__(self, filename, datafor='train'):
        data = pd.read_csv(filename)
        cols =[x for x in data.columns if x not in ['target']]
        rowused = []
        for i in range (len(data)):
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
        data_y = selected_data.loc[:, 'target']
        self.y = le.fit_transform(data_y.values.ravel())
        self.y = self.y.reshape(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (self.X.iloc[index, :].to_numpy(), self.y[index])

spectrum_train = SpectralDataset('SR-FTIR_data.csv', datafor = 'train')
spectrum_test = SpectralDataset('SR-FTIR_data.csv', datafor = 'test')
spectrum_val = SpectralDataset('SR-FTIR_data.csv', datafor = 'validate')
    
train_loader = DataLoader(spectrum_train, batch_size=128, shuffle=True)
test_loader = DataLoader(spectrum_test, batch_size=64, shuffle=True)
valid_loader = DataLoader(spectrum_val, batch_size=64, shuffle=True)

class Model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
            
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
       
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU())  
                
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU()) 
        
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv11 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU())  
                
        self.conv12 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU()) 
        
        self.conv13 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU())
        
        self.conv14 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv15 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU())  
                
        self.conv16 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()) 
        
        self.conv17 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        
        self.conv18 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv19 = nn.Sequential(
            nn.Conv1d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU())  
                
        self.conv20 = nn.Sequential(
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU()) 
        
        self.conv21 = nn.Sequential(
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU())
        
        self.conv22 = nn.Sequential(
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
                
        
        self.flatten = nn.Flatten()
        self.lin_1 = nn.Linear(2048, 512)
        self.lin_2 = nn.Linear(512, 256)
        self.lin_3 = nn.Linear(256, 128)
        self.lin_4 = nn.Linear(128, 25)
        
    def forward(self, x):
        x = x.view(x.shape[0], 1,-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.flatten(x)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = F.relu(self.lin_3(x))
        x = self.lin_4(x)
        return x

model = Model_CNN()
optimizer = optim.Adam(model.parameters(), lr=1e-04)
criterion = nn.CrossEntropyLoss()
    
class MtecDataset:
    def __init__(self, filename):
        data = pd.read_csv(filename)
        cols =[x for x in data.columns if x not in ['target']]
        self.X = data[cols]
        self.y = data.loc[:, 'target']
        self.y = self.y.values.reshape(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return (self.X.iloc[index, :].to_numpy(), self.y[index])


def train(model, optimizer, criterion, train_loader, valid_loader, test_loader, epochs=50, device='cpu'):      
    val_accplot, val_lossplot = [], []
    train_accplot, train_lossplot = [], []
    best_loss = 1
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        valid_loss = 0
        num_train_correct = 0
        num_train_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs, target = inputs.float().to(device), target.to(device)
            target = target.type(torch.LongTensor)
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
            predicted_train = torch.max(F.softmax(output, dim=1), dim=1)[1]
            train_correct = torch.eq(predicted_train, target).view(-1)
            num_train_correct += torch.sum(train_correct).item()
            num_train_examples += train_correct.shape[0]
        training_loss /= len(train_loader)
        train_accuracy = num_train_correct / num_train_examples
           
        model.eval()
        num_val_correct = 0
        num_val_examples = 0
        y_pred, y_true, = [], []
        with torch.no_grad():
            for batch in valid_loader:
                inputs, target = batch
                inputs, target = inputs.float().to(device), target.to(device)
                output = model(inputs)
                target = target.type(torch.LongTensor)
                loss = criterion(output, target)
                valid_loss += loss.data.item()
                y_score = F.softmax(output, dim=1)
                max_pred = torch.max(y_score, 1)[1]
                val_correct = torch.eq(max_pred,target).view(-1)
                num_val_correct += torch.sum(val_correct).item()
                num_val_examples += val_correct.shape[0]
                y_pred.append(y_score.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())
        valid_loss /= len(valid_loader)
        val_accuracy = num_val_correct / num_val_examples
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        print('\nEpoch: {}, Training loss: {:.4f}, Valid loss= {:.4f}, Train accuracy = {:.4f}, Valid accuracy = {:.4f}'
              .format(epoch, training_loss, valid_loss, 100*train_accuracy, 100*val_accuracy))
        train_lossplot.append(training_loss)
        train_accplot.append(train_accuracy)
        val_accplot.append(val_accuracy)
        val_lossplot.append(valid_loss)
    torch.save(model.state_dict(), 'Model_working.pth')        

train(model, optimizer, criterion, train_loader, valid_loader, test_loader)
print(50*'-')                
def test(device='cpu'): 
    model = Model_CNN()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('Model_working.pth'))
    model.eval()
    num_correct = 0
    num_example = 0
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, target = batch
            inputs, target = inputs.float().to(device), target.to(device)
            target = target.type(torch.LongTensor)
            print('\ntarget    :',target)
            output = model(inputs)
            loss = criterion(output, target)
            test_loss += loss.data.item()
            y_score_test = F.softmax(output, dim=1)
            max_pred_test = torch.max(y_score_test, 1)[1]
            print('Predict as:',max_pred_test)
            correct = torch.eq(max_pred_test,target).view(-1)
            print('Accuracy:',correct)
            num_correct += torch.sum(correct).item()
            num_example += correct.shape[0]
    test_loss /= len(test_loader)
    test_accuracy = 100 * num_correct / num_example
    print(f'Model performance for test set is...Test loss: {test_loss}... Test Accuracy: {test_accuracy}')
    
test()

