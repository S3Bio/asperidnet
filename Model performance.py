#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import optuna
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

spectrum_train = SpectralDataset('dataframe.csv', datafor = 'train')
spectrum_test = SpectralDataset('dataframe.csv', datafor = 'test')
spectrum_val = SpectralDataset('dataframe.csv', datafor = 'validate')
    
train_loader = DataLoader(spectrum_train, batch_size=64, shuffle=True)
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
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.conv9 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=1),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.conv10 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))
        
        self.flatten = nn.Flatten()
        self.lin_1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU())
        self.lin_2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(p=0.2),
            nn.ReLU())
        self.lin_3 = nn.Linear(256, 25)
        
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
        x = self.flatten(x)
        x = self.lin_1(x)
        x = self.lin_2(x)
        x = self.lin_3(x)
        return x

model = Model_CNN()
optimizer = optim.Adam(model.parameters(), lr=1e-04)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, criterion, train_loader, valid_loader, test_loader, epochs=40, device='cpu'):      
    val_accplot, val_lossplot = [], []
    train_accplot, train_lossplot = [], []
    best_loss = 3
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
        
        """Save model when found lowest loss"""
        if valid_loss < best_loss:
            print('\nSaving model as aspModel3_10.pth')
            torch.save(model.state_dict(), 'aspModel3_10.pth')
            best_loss = valid_loss
        
        """Collect data for plotting"""
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        print('Epoch: {}, Training loss: {:.4f}, Valid loss= {:.4f}, Train accuracy = {:.4f}, Valid accuracy = {:.4f}'
              .format(epoch, training_loss, valid_loss, 100*train_accuracy, 100*val_accuracy))
        print('--------------------------')
        train_lossplot.append(training_loss)
        train_accplot.append(train_accuracy)
        val_accplot.append(val_accuracy)
        val_lossplot.append(valid_loss)    
    
    """Plot model efficiency"""
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(11,4))
    plt.tight_layout(pad=2, w_pad=2) #add space between edge and fig , 2 subplot
    ax[0].plot(train_accplot, color='teal', label="Train")
    ax[0].plot(val_accplot, color='darkorange', label="Validation")
    ax[0].legend(fontsize=12,markerscale=1.5)
    ax[0].grid(False)
    ax[0].set_title("Training Accuracy", fontsize=14);
    ax[0].set_xlabel("Epoch",fontsize=14);
    ax[0].set_ylabel("Accuracy",fontsize=14);
    ax[1].plot(train_lossplot, color='teal', label="Train")
    ax[1].plot(val_lossplot, color='darkorange',label="Validation")
    ax[1].legend(fontsize=12,markerscale=1.5)
    ax[1].grid(False)
    ax[1].set_title("Training Loss", fontsize=14);
    ax[1].set_xlabel("Epoch",fontsize=14);
    ax[1].set_ylabel("Loss",fontsize=14);
    fig.savefig("aspModel3_10.png", format='png')
    
train(model, optimizer, criterion, train_loader, valid_loader, test_loader)
print('--------------------------------------------------------------------------------------------------------------')
def test(device='cpu'):
    model = Model_CNN()
    model.load_state_dict(torch.load('aspModel3_10.pth'))
    model.eval()
    num_correct = 0
    num_example = 0
    test_loss = 0
    y_pred, y_true, = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, target = batch
            inputs, target = inputs.float().to(device), target.to(device)
            target = target.type(torch.LongTensor)
            output = model(inputs)
            loss = criterion(output, target)
            test_loss += loss.data.item()
            y_score_test = F.softmax(output, dim=1)
            max_pred_test = torch.max(y_score_test, 1)[1]
            correct = torch.eq(max_pred_test,target).view(-1)
            num_correct += torch.sum(correct).item()
            num_example += correct.shape[0]
            y_pred.append(y_score_test.cpu().detach().numpy())
            y_true.append(target.cpu().detach().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
    test_loss /= len(test_loader)
    test_accuracy = 100 * num_correct / num_example
    print(f'Model performance for test set is...Test loss: {test_loss}... Test Accuracy: {test_accuracy}')
    print('--------------------------')
    
    """Compute the Matthews correlation coefficient (MCC)"""
    y_predict = np.argmax(y_pred, axis=-1)
    mcc = matthews_corrcoef(y_true, y_predict)
    print('MCC:',mcc)
    print('--------------------------')
    
    """Compute the classification metrics"""
    class_label = np.arange(1,23)
    print(classification_report(y_true, y_predict, labels=class_label-1))
    print('--------------------------')
    
    """Compute the Confusion_matrix"""
    cm = confusion_matrix(y_true, y_predict)
    
    """Plot confusion matrix"""
    fig = plt.figure(figsize=(10, 8))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, annot_kws={"fontsize":12}, ax = ax,
    cmap='gray_r', vmin=0, vmax=6,
    xticklabels=class_label, yticklabels=class_label,
    linewidths=0.5, linecolor='k', cbar=False)
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=14)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=0)
    ax.xaxis.set_ticklabels(class_label, fontsize = 12)
    ax.xaxis.tick_bottom()
    ax.set_ylabel('True', fontsize=14)
    ax.yaxis.set_ticklabels(class_label, fontsize = 12)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig('ConMat_aspModel3_10.png')
    plt.show()        

test()

