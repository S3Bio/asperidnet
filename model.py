import torch.nn as nn

def createConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


def createConvLayerStack(in_channels, conv_kernel_size=3, conv_stride=1, conv_padding=0, pool_kernel_size=3, pool_stride=3):
    out_channels = in_channels * 2
    return nn.Sequential(
        createConvLayer(in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
        createConvLayer(out_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
        createConvLayer(out_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
        createConvLayer(out_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
        nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.structure = nn.Sequential(
            createConvLayer(1,64, kernel_size=3, stride=1), #1
            createConvLayer(64,64, kernel_size=3, stride=1), #2
            nn.MaxPool1d(kernel_size=3, stride=3),
            createConvLayerStack(64, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3, pool_stride=3), #3-6
            createConvLayerStack(128, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3, pool_stride=3), #7-10
            createConvLayerStack(256, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3, pool_stride=3), #11-14
            createConvLayerStack(512, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3, pool_stride=3), #15-18
            createConvLayerStack(1024, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3, pool_stride=3), #19-22
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 25)
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], 1,-1)
        return self.structure(x)

if __name__ == '__main__':
    model = Model()
    print(model)