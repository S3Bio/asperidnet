import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class SpectralDataset:
    def __init__(self, filename, datafor='train'):
        data = pd.read_csv(filename)
        data.drop_duplicates()

        self.X = data.drop(['target'], axis=1)

        le = LabelEncoder()
        self.y = data.loc[:, 'target']
        le.fit(self.y.values.ravel())

        if datafor == 'train':
            filter = [i % 10 != 0 and i % 10 != 1 for i in range(len(data))]
        elif datafor == 'validate':
            filter = [i % 10 ==  1 for i in range(len(data))]
        elif datafor == 'test':
            filter = [i % 10 ==  0 for i in range(len(data))]
        else:
            filter = [True for i in range(len(data))]
        
        self.X = self.X[filter]
        self.y = self.y[filter]
        self.y = le.transform(self.y.values.ravel())
        self.y = self.y.reshape(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index, :], dtype=torch.float32).view(1,-1), torch.tensor(self.y[index]))
    
if __name__ == '__main__':
    dataset = SpectralDataset('SR-FTIR data.csv', datafor = 'train')

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset)

    for x,y in train_loader:
        # print(x.shape)
        print(y)
        # x = x.view(x.shape[0], 1,-1)
        print(y.shape)
        break
