from neuronCoverage import NeuronCoverage
from deepGauge import DeepGauge
import numpy as np 
import torch
import torch.nn as nn
from model import Epoch, weight_init, build_vgg16, build_resnet101, CNN3D, RNN, Vgg16_diff, EpochSingle, Resnet101_diff
from data import A2D2Dataset,A2D23D, A2D2Diff, A2D2MT
import matplotlib.image as mpimg
import pandas as pd 
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def nbc_test(model_name, model_path,training_set, source_test_set, transform_test_set, dist_file=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "vgg16":
        model = Vgg16_diff()
    elif model_name == "epoch":
        model = Epoch()
    elif model_name == "CNN3D":
        model = CNN3D(sequence_len=2)
    elif model_name == "resnet101":
        model = Resnet101_diff()

    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_name+'-30.pt')) 
    model = model.to(device)
    model.eval()
    dg = DeepGauge(model, 100)
    if not dist_file:
        dg.set_distritubion(training_set)
        dg.save_neuron_distribution("neuron_dist_a2d2_" + model_name)
    else:
        dg.set_distribution_from_file("neuron_dist_a2d2_" + model_name)

    def test_on(test_set):
        for i, data in enumerate(test_set):         
            source_img = data[0]
            source_speed = torch.from_numpy(data[1])
            source_img_tensor = source_img.unsqueeze(0)
            source_img_tensor = source_img_tensor.type(torch.FloatTensor)
            source_img_tensor = source_img_tensor.to(device)
            source_speed = source_speed.unsqueeze(0)
            source_speed = source_speed.type(torch.FloatTensor)
            source_speed = source_speed.to(device)

            dg.update_boundry_coverage((source_img_tensor, source_speed))

    test_on(source_test_set)
    print(dg.get_boundry_coverage())
    for transform_list in transform_test_set:
        for dataset in transform_list:
            test_on(dataset)
        print(dg.get_boundry_coverage())

def nc_test(model_name, model_path, source_test_set, transform_test_set, threshold):
    # print(transform_test_set_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "vgg16":
        model = Vgg16_diff()
    elif model_name == "epoch":
        model = Epoch()
    elif model_name == "CNN3D":
        model = CNN3D(sequence_len=2)
    elif model_name == "resnet101":
        model = Resnet101_diff()

    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_name+'-30.pt')) 
    
    # model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    nc_source = NeuronCoverage(model, threshold=threshold)


    # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def test_on(test_set):
        for i, data in enumerate(test_set):
            source_img = data[0]
            source_speed = torch.from_numpy(data[1])
            source_img_tensor = source_img.unsqueeze(0)
            source_img_tensor = source_img_tensor.type(torch.FloatTensor)
            source_img_tensor = source_img_tensor.to(device)
            source_speed = source_speed.unsqueeze(0)
            source_speed = source_speed.type(torch.FloatTensor)
            source_speed = source_speed.to(device)

            nc_source.update_coverage((source_img_tensor,source_speed))
    test_on(source_test_set)
    print(nc_source.get_coverage())
    test_on(transform_test_set)
    print(nc_source.get_coverage())

train_composed = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

test_composed = transforms.Compose([transforms.ToTensor()])
train_dataset = A2D2Diff(phase="train", transform=test_composed, root_path="..")

source_test_dataset = A2D2Diff(phase="test", transform=test_composed, root_path="..")
transform_test_dataset_1_to_5 = [A2D2MT(root_path="..", source_path="car_30", mt_path="car_30", mode="self"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="day2night", mode="self"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="day2rain", mode="self"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="person_right_side", mode="self"),                      
                        A2D2MT(root_path="..", source_path="car_100", mt_path="bicycle_adjust", mode="self"),                      
]

transform_test_dataset_6_to_7 = [
                        A2D2MT(root_path="..", source_path="car_100", mt_path="car_50", mode="compare"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="add_speed_sign_07", mode="self"),
]

transform_test_dataset_9_to_10 = [A2D2MT(root_path="..", source_path="car_100", mt_path="car_50", mode="compare"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="car_30", mode="compare"),
                        A2D2MT(root_path="..", source_path="car_100", mt_path="add_car2rain", mode="self"),
]

model = ("resnet101", "")
# model = ("")
transforms = [transform_test_dataset_1_to_5 + transform_test_dataset_6_to_7, transform_test_dataset_9_to_10]
# for i, transform_dataset in enumerate(test_datasets):
# nc_test(model[0], model[1], source_test_dataset, transform_test_dataset, 0.25)
nbc_test(model[0], model[1], train_dataset, source_test_dataset, transforms, dist_file=True)
# nbc_test(model[0], model[1], train_dataset, source_test_dataset, transforms + transform_test_dataset_9_to_10, dist_file=True)

    # if i == 0:
    #     nbc_test(model[0], model[1], train_dataset,"source_datasets/original", transform_dataset, dist_file=True)
    # else:
    #     nbc_test(model[0], model[1], train_dataset,"source_datasets/original", transform_dataset, dist_file=True)


