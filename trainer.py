import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import torch

from model import Model
from dataset import SpectralDataset

model = Model()
train_loader = DataLoader(SpectralDataset('SR-FTIR data.csv', datafor = 'train'))
# train model
trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=train_loader)

# for X, y in train_loader:
#     print(X)
#     print(y)
#     print(model(X))
#     break