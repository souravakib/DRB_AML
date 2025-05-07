import cv2
import numpy as np
import os
import torch.utils.data as data
from .transformer import  *
import torchvision.transforms as transforms
import random


class dataset_Aptos(data.Dataset):
    def __init__(self, data_path, DF, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.DF = DF

    def __getitem__(self, index):
        while True:  # retry until valid image found
            image_id = self.DF.loc[index, 'image']
            label = self.DF.loc[index, 'diagnosis']
            label_onehot = np.eye(5)[label]
    
            if not image_id.endswith('.png'):
                image_id += '.png'
    
            img_path = os.path.join(self.data_path, image_id).replace('\\', '/')
    
            Img = cv2.imread(img_path)
            if Img is None:
                print(f"❌ Skipping unreadable image: {img_path}")
                index = random.randint(0, len(self.DF) - 1)  # try another
                continue
    
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)
    
            if self.transform is not None:
                Img = self.transform(Img)
    
            return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)


class dataset_RFMiD(data.Dataset):
    def __init__(self, data_path, DF, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.DF = DF

    def __getitem__(self, index):
        image_id = self.DF.loc[index, 'image']  # or 'ID' depending on your CSV
        label = self.DF.loc[index, 'Disease_Risk']  # check this column name too

        if not image_id.endswith('.png'):
            image_id += '.png'

        img_path = os.path.join(self.data_path, image_id).replace('\\', '/')
        Img = cv2.imread(img_path)

        if Img is None:
            raise FileNotFoundError(f"❌ Image not found or unreadable: {img_path}")

        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = transforms.ToPILImage()(Img)

        if self.transform is not None:
            Img = self.transform(Img)

        return Img, label

    def __len__(self):
        return len(self.DF)

class dataset_DR(data.Dataset):
    def __init__(self, data_path, DF, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.DF = DF

    def __getitem__(self, index):
        while True:  # retry until valid image found
            image_id = self.DF.loc[index, 'image']
            label = self.DF.loc[index, 'diagnosis']
            label_onehot = np.eye(5)[label]
    
            if not image_id.endswith('.png'):
                image_id += '.png'
    
            img_path = os.path.join(self.data_path, image_id).replace('\\', '/')
    
            Img = cv2.imread(img_path)
            if Img is None:
                print(f"❌ Skipping unreadable image: {img_path}")
                index = random.randint(0, len(self.DF) - 1)  # try another
                continue
    
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)
    
            if self.transform is not None:
                Img = self.transform(Img)
    
            return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)


