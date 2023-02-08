import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import torch

from model import Model
from dataset import SpectralDataset

model = Model()
train_loader = DataLoader(SpectralDataset('SR-FTIR data.csv', datafor = 'train'))
# train model
# trainer = pl.Trainer()
# trainer.fit(model=model, train_dataloaders=train_loader)
# model = nn.Sequential(
#     nn.Sequential(
#         nn.Conv1d(1, 64, kernel_size=3, stride=1),
#         nn.BatchNorm1d(64),
#         nn.ReLU()
#     ),
#     nn.Sequential(
#         nn.Conv1d(64, 64, kernel_size=3, stride=1),
#         nn.BatchNorm1d(64),
#         nn.ReLU(),
#         nn.MaxPool1d(kernel_size=3, stride=3)
#     )
# )

for X, y in train_loader:
    print(X)
    print(y)
    print(model(X))
    break