import pandas as pd
from sklearn.preprocessing import LabelEncoder

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