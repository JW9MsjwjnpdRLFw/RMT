import numpy as np 
import torch
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init, build_resnet101, build_resnet
import matplotlib.image as mpimg
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mt_checker import mt_distribution


def generate_csv():
    # pairs = [('day2rain', 'day2rain'), ('day2snow', 'day2snow'), ('gen_car', 'add_car2night'), ('gen_person', 'add_person2night'), ('gen_bicycle', 'add_bicycle2night'), ('original','dt_rotation')]
    pairs = [('original', 'DT_Shear'), ('original', 'DT_Rotation'), ('original', 'DT_Contrast'), ('original', 'DT_Brightness'), ('original', 'DT_Translation'), ('original', 'DT_Blur')]
    names = ['Adding a car', 'Adding a bicycle', 'Adding a pedestrian', 'Day2night', 'Original2rainy']
    model_names = ['basecnn', 'vgg16', 'resnet101']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_dataset_dir = 'source_datasets'
    follow_up_dataset_dir = 'follow_up_datasets'

    for model_name in model_names:
        if model_name == 'basecnn':
            model = BaseCNN()
            model.load_state_dict(torch.load('models/basecnn.pt'))
        elif model_name == 'vgg16':
            model = build_vgg16(False)
            model.load_state_dict(torch.load('models/vgg16.pt'))
        elif model_name == 'resnet101':
            model = build_resnet101(False)
            model.load_state_dict(torch.load('models/resnet101.pt'))
        model = model.to(device)
        model.eval()
        for (source, follow) in pairs:
            print(model_name, follow)
            img_list = os.listdir(os.path.join(follow_up_dataset_dir, follow))
            preds = {"name": [], "ori_pred": [], "mod_pred": []}
            for img_name in img_list:
                source_img = mpimg.imread(os.path.join(source_dataset_dir, source, img_name))
                follow_img = mpimg.imread(os.path.join(follow_up_dataset_dir, follow, img_name))  
                if np.max(source_img) > 1:
                    source_img = source_img / 255.
                if np.max(follow_img) > 1:
                    follow_img = follow_img / 255.   
                source_img_tensor = torch.from_numpy(np.transpose(source_img, (-1, 0, 1))).unsqueeze(0)
                source_img_tensor = source_img_tensor.type(torch.FloatTensor)
                source_img_tensor = source_img_tensor.to(device)
                follow_img_tensor = torch.from_numpy(np.transpose(follow_img, (-1, 0, 1))).unsqueeze(0)
                follow_img_tensor = follow_img_tensor.type(torch.FloatTensor)
                follow_img_tensor = follow_img_tensor.to(device)  

                ori_pred = model(source_img_tensor)
                mod_pred = model(follow_img_tensor)
                preds["name"].append(img_name)
                preds["ori_pred"].append(ori_pred.item())
                preds["mod_pred"].append(mod_pred.item())

            preds = pd.DataFrame(preds)
            preds.to_csv("result_" + follow + "_" + model_name + ".csv")

def generate_sensitivity_plot():
    transforms = ["day2rain", "day2snow", "add_car2night", "add_person2night", "add_bicycle2night", "DT_Shear", "DT_Rotation", "DT_Contrast", "DT_Brightness", "DT_Translation", "DT_Blur"]
    for transform in transforms:
        epoch_df = pd.read_csv("result_" + transform + "_basecnn.csv")
        vgg16_df = pd.read_csv("result_" + transform + "_vgg16.csv")
        resnet101_df = pd.read_csv("result_" + transform + "_resnet101.csv")
        mt_distribution(epoch_df, vgg16_df, resnet101_df, transform)

if __name__ == "__main__":
    generate_sensitivity_plot()

