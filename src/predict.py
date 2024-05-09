import cv2
import torch.nn as nn
from osgeo import gdal
import torch
import numpy as np

class UNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(input_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        
        center = self.center(self.pool(encoder4))
        
        decoder4 = self.decoder4(torch.cat([encoder4, self.up(center)], 1))
        decoder3 = self.decoder3(torch.cat([encoder3, self.up(decoder4)], 1))
        decoder2 = self.decoder2(torch.cat([encoder2, self.up(decoder3)], 1))
        decoder1 = self.decoder1(torch.cat([encoder1, self.up(decoder2)], 1))
        
        final = self.final(decoder1).squeeze()
        
        return torch.sigmoid(final)

if __name__ == '__main__':
    model = UNet(3, 1)
    model.load_state_dict(torch.load('E:/Codes/RS/Codes/RS_Segmentation/model.pth'))
    model.eval()
    
    dataset_path = "E:/data/data_building/"
    image_file = dataset_path + 'data/2_955_sat.tif'
    mask_file = dataset_path + 'data/2_955_mask.tif'
    
    mask_data = gdal.Open(mask_file)
    mask_images = mask_data.GetRasterBand().ReadAsArray()
    
    data = gdal.Open(image_file)
    if data is None:
        print("Failed to open the TIFF file.")
        exit()
        
    images = (np.stack([data.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
    test_images = torch.tensor(images).float().unsqueeze(0)
    outputs = model(test_images)
    outputs = (outputs > 0.8).float()
    
    cv2.imshow('Prediction', outputs.numpy())
    cv2.imshow('Label', mask_images)
    cv2.waitKey(0)