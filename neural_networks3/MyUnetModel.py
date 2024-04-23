import torch
import random
from torch import nn
from matplotlib import pyplot as plt

class MyUnetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, desc_level=5):
        super().__init__()
        if (desc_level < 1):
            raise Exception(f"desc_level must be >= 1, got{desc_level}")
        self.desc_level = desc_level
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.asc_conv1 = self._asc_layer(in_channels, 32)
        self.asc_conv2 = self._asc_layer(32, 64)
        self.asc_conv3 = self._asc_layer(64, 128)
        self.asc_conv4 = self._asc_layer(128, 256)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck_layer = nn.Sequential(
            self._conv_block(256, 512),
            self._conv_block(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )

        

        self.desc_conv1 = self._desc_layer(512, 256)
        self.desc_conv2 = self._desc_layer(256, 128)
        self.desc_conv3 = self._desc_layer(128, 64)
        self.desc_conv4 = self._desc_layer(64, 32)
        self.desc_conv5 = self._desc_layer(32, self.out_channels)

        self.last_layer = nn.Sequential(
            self._conv_block(32, 32),
            nn.Conv2d(32, self.out_channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        asc_out1 = self.asc_conv1(x)
        #print("___")
        #print(asc_out1)
        #print("___")
        asc_out2 = self.asc_conv2(self.maxpool(asc_out1))
        asc_out3 = self.asc_conv3(self.maxpool(asc_out2)) 
        asc_out4 = self.asc_conv4(self.maxpool(asc_out3)) 
        bottleneck = self.bottleneck_layer(self.maxpool(asc_out4))
        #print((self.desc_conv2(bottleneck).shape, bottleneck.shape)) 
        desc_out4 = self.desc_conv1(torch.concat((bottleneck, asc_out4), dim=1))  
        desc_out3 = self.desc_conv2(torch.concat((asc_out3, desc_out4), dim=1))
        desc_out2 = self.desc_conv3(torch.concat((asc_out2, desc_out3), dim=1))
        #desc_out2 = 
        
        out = self.last_layer(desc_out2)
        
        #out = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)(desc_out2)
        #out = nn.Sigmoid()(out)
        return out


    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _asc_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self._conv_block(in_channels, out_channels),
            self._conv_block(out_channels, out_channels),
            
        )

    def _desc_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self._conv_block(in_channels, in_channels//2),
            self._conv_block(in_channels//2,in_channels//2),
            nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=2, stride=2)
        )

if __name__ == "__main__":
    example = torch.ones(size=(1, 3, 512, 512))
    model = MyUnetModel(out_channels=2)
    model.eval()
    result1 = model(example).argmax(1).squeeze()
    print(result1.shape)
