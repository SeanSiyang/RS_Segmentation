import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import numpy as np
import torch.nn as nn

class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
      self.images = self.read_multiband_images(images_dir)
      self.labels = self.read_singleband_labels(labels_dir)
      
    def read_multiband_images(self, images_dir):
        images = []
        for image_path in images_dir:
            rsdl_data = gdal.Open(image_path)
            images.append(np.stack([rsdl_data.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
        return images
    
    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_path in labels_dir:
            rsdl_data = gdal.Open(label_path)
            labels.append(rsdl_data.GetRasterBand(1).ReadAsArray())
        return labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)
    
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
    print(torch.cuda.is_available())
    print(gdal.__version__)
    
    dataset_path = "E:/data/data_building/"
    
    images_dir = [dataset_path + 'data/2_95_sat.tif', dataset_path + 'data/2_96_sat.tif',  dataset_path + 'data/2_97_sat.tif', 
                 dataset_path + 'data/2_98_sat.tif', dataset_path + 'data/2_976_sat.tif']
    labels_dir =[dataset_path + 'data/2_95_mask.tif', dataset_path + 'data/2_96_mask.tif',  dataset_path + 'data/2_97_mask.tif', 
                 dataset_path + 'data/2_98_mask.tif', dataset_path +'data/2_976_mask.tif']

    dataset = RSDataset(images_dir, labels_dir)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = UNet(3, 1)
    
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = labels.float() / 255.0
            labels = labels.squeeze(0)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))   
        
    torch.save(model.state_dict(), 'model.pth')
