import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class BaseNet(nn.Module):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),#5x256x50
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#5x128x50
            # Defining another 2D convolution layer
            nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1),#1x128x50
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#1x64x50
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3200, 160),
            nn.Linear(160, 21 * 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)#n_batch x 
        x = self.linear_layers(x)
        return x

class LSTMNet(nn.Module):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),#7x256x5 -> 5x256x5
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#5x128x5
            # Defining another 2D convolution layer
            nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1),#1x128x5
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#1x64x5
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(320, 160),
            nn.Linear(160, 21 * 3)
        )

        self.lstm = nn.LSTM(320, 320, bidirectional=False, batch_first=True)

    def forward(self, x):
        batch_size, nchannels, nshift, window = x.shape # 128, 7, 256, 50
        x = x.transpose(3, 1) # 128, 50, 7, 256
        x = x.reshape(batch_size, 10, 5, nchannels, nshift) # 128, 10, 5, 7, 256
        x = x.permute(0, 1, 3, 4, 2) # 128, 10, 7, 256, 5
        x = x.reshape(-1, nchannels, nshift, 5) # 1280, 7, 256, 5
        x = self.cnn_layers(x) # 1280, 1, 64, 5
        x = x.reshape(x.shape[0], -1) #1280, 320
        x = x.reshape(batch_size, 10, -1) #128, 10, 320
        x = self.lstm(x)[0] # 128, 10, 320
        x = x[:,-1,:] # 128, 320
        x = self.linear_layers(x) #128, 63
        return x
    

class LSTMNetSimple(nn.Module):   
    def __init__(self):
        super().__init__()

        self.Bn = nn.BatchNorm2d(7)
        self.Ln = nn.LayerNorm([7, 256, 50], elementwise_affine=False)
        self.In = nn.InstanceNorm2d(7)
        self.reduction_layers = nn.Sequential(
            nn.Linear(256*7, 320),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 21*3),
        )

        self.lstm = nn.LSTM(320, 320, bidirectional=False, batch_first=True)

    def forward(self, x):
        batch_size, nchannels, nshift, window = x.shape # 128, 7, 256, 50
        #x = self.Bn(x)
        x = self.Ln(x)
        #x = self.In(x)
        x = x.transpose(3, 1) # 128, 50, 7, 256
        x = x.reshape(batch_size, window, -1) #128, 10, 320
        x = self.reduction_layers(x)
        x = self.lstm(x)[0] # 128, 10, 320
        x = x[:,-1,:] # 128, 320
        x = self.linear_layers(x) #128, 63
        return x

    
class LSTMAngleNet(nn.Module):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),#5x256x5
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#5x128x5
            # Defining another 2D convolution layer
            nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1),#1x128x5
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),#1x64x5
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(320, 160),
            nn.Linear(160, 19)
        )

        self.lstm = nn.LSTM(320, 320, bidirectional=False, batch_first=True)

    def forward(self, x):
        batch_size, nchannels, nshift, window = x.shape # 128, 7, 512, 50
        x = x.transpose(3, 1) # 128, 50, 7, 512
        x = x.reshape(batch_size, 10, 5, nchannels, nshift) # 128, 10, 5, 7, 512
        x = x.permute(0, 1, 3, 4, 2) # 128, 10, 7, 512, 5
        x = x.reshape(-1, nchannels, nshift, 5)
        x = self.cnn_layers(x) # 1280, 1, 128, 5
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, 10, -1)
        x = self.lstm(x)[0] # 128, 10, 640
        x = x[:,-1,:] # 128, 640
        x = self.linear_layers(x)
        return x

class ResNet(nn.Module):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(7, 3, kernel_size=1),#5x512x5
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.transform = nn.Linear(512*5, 30*30) # 30 depends on self.cnn_layers

        self.linear_layers = nn.Sequential(
            nn.Linear(1000, 160),
            nn.Linear(160, 21 * 3)
        )
        self.resnet = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        bs, nc, dim, sl = x.shape
        x = self.transform(x.reshape(bs,nc,-1)).reshape(bs,nc,30,30) # 30 depends on self.cnn_layers
        x = self.cnn_layers(x)
        x = self.resnet(x)
        x = x.reshape(x.size(0), -1)#n_batch x 
        x = self.linear_layers(x)
        return x

class RLNet(nn.Module):   # resnet with lstm
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(7, 3, kernel_size=1),#5x512x5
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.transform = nn.Linear(512*5, 30*30) # 30 depends on self.cnn_layers

        self.linear_layers = nn.Sequential(
            nn.Linear(1000, 160),
            nn.Linear(160, 21 * 3)
        )
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.lstm = nn.LSTM(1000, 1000, bidirectional=False, batch_first=True)

    def forward(self, x):
        batch_size, nchannels, nshift, window = x.shape # 128, 7, 512, 50
        x = x.transpose(3, 1) # 128, 50, 7, 512
        x = x.reshape(batch_size, 10, 5, nchannels, nshift) # 128, 10, 5, 7, 512
        x = x.permute(0, 1, 3, 4, 2) # 128, 10, 7, 512, 5
        x = x.reshape(-1, nchannels, nshift, 5) # 1280, 7, 512, 5
        
        x = self.transform(x.reshape(batch_size*10,nchannels,-1)).reshape(batch_size*10,nchannels,30,30) # 30 depends on self.cnn_layers
        
        x = self.cnn_layers(x) # 1280, 1, 128, 5
        
        x = self.resnet(x)
        
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, 10, -1)
        x = self.lstm(x)[0] # 128, 10, 640
        x = x[:,-1,:] # 128, 640
        x = self.linear_layers(x)
        return x
