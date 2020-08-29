import matplotlib
matplotlib.use('Agg')
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init, build_resnet101, build_resnet
from data import CityScapesDataset
import torch.optim as optim
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import csv
from os import path
# from scipy.misc import imread, imresize, imsave
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
from torch.optim import lr_scheduler

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models")
    parser.add_argument('--model_name', action='store', type=str, required=True)
    parser.add_argument('--data_root', action='store', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',  type=int, default=20)
    parser.add_argument('--re_train', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    re_train = args.re_train
    test = args.test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    if model_name == 'epoch':
        model = BaseCNN()
    elif model_name == 'vgg16':
        model = build_vgg16(False)
    elif model_name == 'resnet101':
        model = build_resnet101(False)
    data_root = args.data_root

    model.apply(weight_init)
    model = model.to(device)
    # transforms.ToPILImage(), transforms.RandomHorizontalFlip(),  , transforms.RandomAffine(degrees=0, translate=(0.1, 0))
    train_composed = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=5, translate=(0.2, 0.2)), transforms.ToTensor()])
    train_dataset = CityScapesDataset(phase='train', transforms=train_composed, img_size=(224, 224), root_path=data_root)
    test_composed = transforms.Compose([transforms.ToTensor()])
    test_dataset = CityScapesDataset(phase='val', transforms=test_composed, img_size=(224, 224), root_path=data_root)
    print(len(train_dataset), len(test_dataset))
    steps_per_epoch = int(len(train_dataset) / batch_size)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.L1Loss()
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.Adam(params_to_update, lr=0.0001)
    # optimizer = optim.Adam(
    #     [
    #         {"params": model.classifier.parameters(), "lr": 1e-3},
    #     ],
    #     lr=1e-4,
    # )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if not test:
        if re_train:

            model.load_state_dict(torch.load('model.pt'))
            model = model.to(device)
        

        

        best = 10
        for epoch in range(epochs):
            exp_lr_scheduler.step()
            model.train()
            total_loss = 0
            for step, sample_batched in enumerate(train_generator):
                if step <= steps_per_epoch:
                    batch_x = sample_batched[0]
                    # print(batch_x.numpy())
                    batch_y = sample_batched[1]

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x).view(-1)
                    print("+", end="")
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                else:
                    break
            print()

            test_loss = 0
            model.eval()
            with torch.no_grad():
                for i, sample_batched in enumerate(test_generator):
                    batch_x = sample_batched[0]
                    # print(batch_x.numpy())
                    batch_y = sample_batched[1]

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x).view(-1)
                    print("-|", end="")
                    # print(torch.mean(outputs), torch.min(outputs), torch.max(outputs))
                    loss = criterion(outputs, batch_y)
                    # optimizer.zero_grad()

                    # loss.backward()
                    # optimizer.step()
                    running_loss = loss.item()
                    test_loss += running_loss

            print()
            print(torch.mean(outputs), torch.min(outputs), torch.max(outputs))
            print('Epoch %d  training RMSE loss: %.4f test loss: %.4f' % (epoch,  total_loss / steps_per_epoch, test_loss / i))
            if test_loss / i < best:
                torch.save(model.state_dict(), 'model.pt')
                best = test_loss / i

        model.load_state_dict(torch.load('model.pt'))
        
    else:
        model.load_state_dict(torch.load('model - ' + model_name + '.pt'))
        print(model)
        # model.load_state_dict(torch.load('vgg16.pt'))

        model = model.to(device)

    test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = model.to(device)    
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, sample_batched in enumerate(test_generator):
            batch_x = sample_batched[0]
            # print(batch_x.numpy())
            batch_y = sample_batched[1]

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            # print(outputs.item(), batch_y.item())
            y_pred.append(outputs.item())
            y_true.append(batch_y.item())
            # loss = criterion(outputs, batch_y)
            # optimizer.zero_grad()
            img = batch_x.squeeze(0).detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            # plt.imshow(img)
            # plt.title('y_true: %.4f y_pred: %.4f' % (batch_y.item(), outputs.item()))
            # plt.savefig('original_prediction/' + str(i))
            # loss.backward()
            # optimizer.step()
            # running_loss = loss.item()
            # test_loss += running_loss

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        f = pd.DataFrame(y_true, columns=['label'])
        f.to_csv('test_set_label.csv')    

        print(mae)