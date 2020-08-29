import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms  
import pandas as pd
import os 
import torch
from PIL import Image
import os
import json
import cv2
import numpy as np
from scipy.misc import imread, imresize, imsave

class CityScapesDataset(Dataset):
    # root_path: store Cityscapes dataset, which includes three folders 'train', 'val', and 'vehicle'
    def __init__(self, phase='train', transforms=None, img_size=(224, 224), root_path = '..'):
        self.root_path = root_path
        self.phase = phase
        self.image_list, self.label_list = self.get_data_list()
        self.transforms = transforms
        self.img_size = img_size
    def get_data_list(self):
        if self.phase == 'train':
            phases = ['train']
        else:
            phases = ['val']
        image_list = []
        label_list = []

        for phase in phases:
            image_directory = os.listdir(os.path.join(self.root_path, phase))
            for directory in image_directory:
                images = os.listdir(os.path.join(self.root_path, phase, directory))
                for img in images:
                    image_list.append((phase, directory, img))
        for phase in phases:
            for img in image_list:
                json_address = os.path.join(self.root_path, 'vehicle', img[0], img[1], img[2][:-15] + 'vehicle.json')
                label_list.append(json_address)
        
        return image_list, label_list

    def get_image(self, img_address):
        img = mpimg.imread(img_address)
        if self.img_size == (224, 224) or self.img_size == (128, 128):
            cropped = img[0:852, 384:1520]
            img = imresize(cropped, self.img_size)
        elif self.img_size == (132, 400):
            cropped = img[144:820, :]
            img = imresize(cropped, (132, 400))
        elif self.img_size == (128, 256):
            img = imresize(img, (128, 256))
        elif self.img_size == (256, 256):
            cropped = img[0:852, 384:1520]
            img = imresize(cropped, (256, 256))

        return img

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_address = os.path.join(self.root_path, self.image_list[idx][0], self.image_list[idx][1], self.image_list[idx][2])
        img = self.get_image(img_address)
        # img = np.transpose(img, (-1, 0, 1))
        with open(self.label_list[idx],'r') as load_f:
            label_json = json.load(load_f)
            label = label_json['speed']
        if self.transforms:
            img = self.transforms(img)

        return img, label
# dataset = CityScapesDataset()
# for i in range(len(dataset)):
#     if i > 1000:
#         img = dataset[i][0]
#         imsave('day/' + str(i) + '.jpg', img)
# print(dataset[1][0].shape, dataset[0][1])
# plt.imshow(np.transpose(dataset[1][0], (1, 2, 0)))
# plt.show()

class DT_Rotation(object):
    def __init__(self, param):
        self.param = param
    
    def __call__(self, sample):
        rows, cols, ch = sample.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.param, 1)
        dst = cv2.warpAffine(sample, M, (cols, rows))
        return dst

class DT_Shear(object):
    def __init__(self, param):
        self.param = param
    
    def __call__(self, sample):
        rows, cols, ch = sample.shape
        factor = self.param*(-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(sample, M, (cols, rows))
        return dst 

class DT_Scale(object):
    def __init__(self, param):
        self.param = param
    
    def __call__(self, sample):
        res = cv2.resize(sample, None, fx=self.param[0], fy=self.param[1], interpolation=cv2.INTER_CUBIC)
        return res  

class DT_Contrast(object):
    def __init__(self, param):
        self.param = param

    def __call__(self, sample):
        alpha = self.param
        new_img = cv2.multiply(sample, np.array([alpha]))                    # mul_img = img*alpha
        #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img              

class DT_Translation(object):
    def __init__(self, param):
        self.param = param
    
    def __call__(self, sample):
        rows, cols, ch = sample.shape

        M = np.float32([[1, 0, self.param[0]], [0, 1, self.param[1]]])
        dst = cv2.warpAffine(sample, M, (cols, rows))
        return dst        

class DT_Brightness(object):
    def __init__(self, param):
        self.param = param
    
    def __call__(self, sample):
        beta = self.param
        new_img = cv2.add(sample, beta)                                  # new_img = img*alpha + beta

        return new_img        

class DT_Blur(object):
    def __init__(self, param):
        self.param = param 
    
    def __call__(self, sample):
        blur = []
        if self.param == 1:
            blur = cv2.blur(sample, (3, 3))
        if self.param == 2:
            blur = cv2.blur(sample, (4, 4))
        if self.param == 3:
            blur = cv2.blur(sample, (5, 5))
        if self.param == 4:
            blur = cv2.GaussianBlur(sample, (3, 3), 0)
        if self.param == 5:
            blur = cv2.GaussianBlur(sample, (5, 5), 0)
        if self.param == 6:
            blur = cv2.GaussianBlur(sample, (7, 7), 0)
        if self.param == 7:
            blur = cv2.medianBlur(sample, 3)
        if self.param == 8:
            blur = cv2.medianBlur(sample, 5)
        if self.param == 9:
            blur = cv2.blur(sample, (6, 6))
        if self.param == 10:
            blur = cv2.bilateralFilter(sample, 9, 75, 75)
        return blur        