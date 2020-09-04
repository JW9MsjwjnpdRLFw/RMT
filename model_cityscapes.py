import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math
import argparse
import os
import matplotlib.image as mpimg
import numpy as np

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(32),
            #nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(64),

            #nn.Dropout(0.25)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.InstanceNorm2d(128),

            #nn.Dropout(0.25)

        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25)

        )
        self.layer5 = nn.Sequential(
            nn.Linear(14*14*256, 1024),
            # nn.Linear(25*8*128, 1024),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        return out

class Nvidia(nn.Module):
    def __init__(self):
        super(Nvidia, self).__init__()
        # 3*66*200
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU(), 
        )
        # 24*31*98
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU(), 
        )
        # 36*14*47
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU(), 
        )
        # 48*5*22
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
        )
        # 64*3*20
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
        )
        # 64*1*18
        self.layer6 = nn.Sequential(
            nn.Linear(64*9*43, 1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        # self.layer7 = nn.Tanh()
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer6(out)
        return out

class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        out = self.model.features(x)
        out = self.conv_new(out)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)

        return out

def build_vgg16(pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    if pretrained:
        # for name, child in model.features.named_children():
        #     # print(name)
        #     if int(name) <= 24:
        #         for params in child.parameters():
        #             params.requires_grad = False
        for parma in model.parameters():
            parma.requires_grad = False    
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(25088),
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )
    return model

def build_resnet(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        for parma in model.parameters():
            parma.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=2048, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=2048, out_features=1, bias=True),
                         )
    
    return model

def build_resnet101(pretrained=False):
    model = models.resnet101(pretrained=pretrained)
    if pretrained:
        for parma in model.parameters():
            parma.requires_grad = False    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=2048, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=2048, out_features=1, bias=True),
                         )
    return model  
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="epoch")
    parser.add_argument("--model_path", type=str, default="../models")
    parser.add_argument("--input_path",  '-i', type=str, help="Path to img for prediction", default="D:/udacity-data/testing/center")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model_name = args.model_name
    model_path = args.model_path
    print(model_name, model_path, args.input_path)
    model = None
    if model_name == 'epoch':
        model = BaseCNN()
    elif model_name == 'vgg16':
        model = build_vgg16(False)
    elif model_name == 'resnet101':
        model = build_resnet101(False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_path = args.input_path
    img_list = os.listdir(img_path)
    predictions = []
    for img_name in img_list:
        img = mpimg.imread(os.path.join(img_path, img_name))
        if np.max(img) > 1:
            img = img / 255.
        img_tensor = torch.from_numpy(np.transpose(img, (-1, 0, 1))).unsqueeze(0)
        img_tensor = img_tensor.type(torch.FloatTensor)
        img_tensor = img_tensor.to(device)
        predictions.append(model(img_tensor).item())
    print(predictions)
# net = BaseCNN()
# net = net.to(device)
# summary(net, (3,224,224))