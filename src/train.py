import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import numpy as np

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

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(gdal.__version__)
    
    dataset_path = "E:/data/data_building/"
    
    images_dir = [dataset_path + 'data/2_95_sat.tif', dataset_path + 'data/2_96_sat.tif',  dataset_path + 'data/2_97_sat.tif', 
                 dataset_path + 'data/2_98_sat.tif', dataset_path + 'data/2_976_sat.tif']
    labels_dir =[dataset_path + 'data/2_95_mask.tif', dataset_path + 'data/2_96_mask.tif',  dataset_path + 'data/2_97_mask.tif', 
                 dataset_path + 'data/2_98_mask.tif', dataset_path +'data/2_976_mask.tif']

    dataset = RSDataset(images_dir, labels_dir)