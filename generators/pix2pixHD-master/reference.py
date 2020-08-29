import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import copy
import numpy as np
import cv2
from scipy.misc import imsave, imresize
import random
import matplotlib.pyplot as plt

def generate_reference(dataset, img_name, position_x, position_y):
    for data in dataset:
        print(img_name, data['path'][0])
        if img_name not in data['path'][0]:
            continue
        else:
            instance_number = data['inst'][0,0, position_x, position_y]
            print(instance_number)
            refernce = (data['inst'] == instance_number).nonzero().numpy()
            np.save('reference.npy', refernce)
            break

def check_reference(reference_name):
    reference = np.load(reference_name)
    img = np.zeros((512, 1024))
    for point in reference:
        img[point[2], point[3]] = 1
    # print(reference)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    # opt.phase = 'train'
        # add instance_feat to control image generation
    opt.instance_feat = True
    # opt.use_encoded_image = True
    # person = np.load('person.npy')
    # person[:, 3] += 100
    # np.save('person.npy', person)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    generate_reference(dataset, opt.img_name, opt.position_x, opt.position_y)
    # check_reference('reference.npy')