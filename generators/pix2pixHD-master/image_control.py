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

def sample_features(inst): 
    # read precomputed feature clusters 
    cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)  
    # features_clustered里存储了cityscape 35个类别里每个instance的特征，因此可以控制generator生成每个instance的样子      
    features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

    # randomly sample from the feature clusters
    inst_np = inst.cpu().numpy().astype(int)                                      
    feat_map = torch.Tensor(inst.size()[0], opt.feat_num, inst.size()[2], inst.size()[3])
    for i in np.unique(inst_np):    
        label = i if i < 1000 else i//1000
        if label in features_clustered:
            feat = features_clustered[label]
            cluster_idx = np.random.randint(0, feat.shape[0]) 
                                        
            idx = (inst == int(i)).nonzero()
            for k in range(opt.feat_num):                                    
                feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
    if opt.data_type==16:
        feat_map = feat_map.half()
    return feat_map

# Cityscapes class: 6: ground; 7: road; 11: building; 24: person; 26: car, 33: bicycle
def remove_object(feat_map, data, region, modify_type=7):
    # cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)        
    # features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

    source_region = (data['inst'] == modify_type).nonzero()
    source_feat = feat_map[source_region[0][0], :, source_region[0][2], source_region[0][3]]
    
    for point in region:
        data['inst'][point[0], point[1], point[2], point[3]] = modify_type
        data['label'][point[0], point[1], point[2], point[3]] = modify_type
        feat_map[point[0], :, point[2], point[3]] = source_feat
    
    return feat_map, data

def add_object(feat_map, data, source_file, source_type=26):
    cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)        
    features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

    # get a random feature of adding objects from feature clusters
    source_feats = features_clustered[source_type]
    cluster_idx = np.random.randint(0, source_feats.shape[0])
    car_instances = []
    for i in np.unique(data['inst'].numpy().astype(int)):
        if i >= source_type * 1000 and i < (source_type + 1) * 1000:
            car_instances.append(i)
    
    car_instances.sort()
    if len(car_instances) == 0:
        car_instances.append(0)
    # scales = [0.8, 0.9 ,1, 1.1]
    # level = random.randint(0, 3)
    source_object = resize_object(source_file, 1)
    # source_object = np.load(source_file)
    # source_object = (source_object == 1).nonzero() 
    # source_object = np.array(source_object)
    # source_object[:, 3] += 20
    # source_object[:, 2] += 50
    x_min = source_object[0].min()
    x_max = source_object[0].max()
    y_min = source_object[1].min()
    y_max = source_object[1].max()
    half_object = (x_min + x_max) // 2
    for i in range(half_object, x_max + 1):
        for j in range(y_min, y_max + 1):
            # collision detection
            if data['label'][0, 0, i, j] == 26 or data['label'][0, 0, i, j] == 24:
                return False, 0, 0

    for i in range(len(source_object[0])):
        feat_map[0, :, source_object[0][i], source_object[1][i]] = torch.Tensor(source_feats[cluster_idx])
    # feat_map[0, 1, source_object[0], source_object[1]] = torch.Tensor(source_feats[cluster_idx][1])
    # feat_map[0, 2, source_object[0], source_object[1]] = torch.Tensor(source_feats[cluster_idx][2])

    data['label'][0, 0, source_object[0], source_object[1]] = source_type
    data['inst'][0, 0, source_object[0], source_object[1]] = torch.tensor(car_instances[-1] + 1)

    return True, feat_map, data

def resize_object(name, scale):
    region = np.load(name)
    if scale != 1:
        img = np.zeros((512, 1024))
        min_row = np.min(region[:, 2])
        min_col = np.min(region[:, 3])
        region[:, 2] -= min_row
        region[:, 3] -=min_col
        img[region[:, 2], region[:, 3]] = 1
        img = cv2.resize(img, None, fx=scale, fy=scale)
        region = (img == 1).nonzero()
        region = np.array(region)
        if scale >= 1.5:
            scale_x = scale_x - 0.3
        if scale < 1:
            min_col -= 100 * (1 - scale)
        else:
            min_col += 100 * (scale - 1)
            
        min_row = int(min_row * scale)
        # min_col = int(min_col * scale_y)
        region[0] += int(min_row)
        region[1] += int(min_col)
    else:
        region = np.array([region[:, 2].reshape(-1), region[:,3].reshape(-1)])
    return region

def save_image(img, address, name):
    img = img.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[0:426, 192:760]
    img = imresize(img, (224, 224))
    imsave(os.path.join(address, name), img)
    # img = cv2.resize(img, (224, 224))*255
    # cv2.imwrite(os.path.join(address, name), img)




if __name__ == "__main__":
    parser = TestOptions()
    parser.parser.add_argument('--dataset_path', type=str, default='../../follow_up_datasets')
    parser.parser.add_argument('--output_path', type=str, default='../../follow_up_datasets')
    parser.parser.add_argument('--feature', type=str, default='bicycle.npy')
    parser.parser.add_argument('--feature_ex', type=str, default='rider.npy')
    # parser.parser.add_argument('--x_n', type=str, default='x_n2')


    opt = parser.parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataroot = opt.dataset_path
    # add instance_feat to control image generation
    opt.instance_feat = True
    # opt.use_encoded_image = True
    # person = np.load('person.npy')
    # person[:, 3] += 100
    # np.save('person.npy', person)

    # opt.dataroot = "../../source_datasets"
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # print(opt.add_object, opt.dataroot, opt.output_path, opt.feature, opt.feature_ex)

    # create website
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)
    source_data_path = os.path.join(os.path.split(opt.output_path)[0], 'x_n1')
    follow_up_data_path = os.path.join(os.path.split(opt.output_path)[0], 'x_n2')

    if not os.path.exists(source_data_path):
        os.makedirs(source_data_path)
    if not os.path.exists(follow_up_data_path):
        os.makedirs(follow_up_data_path)
    file_name = 0
    for i, data in enumerate(dataset):
        if i < 500:
        #     break
        # new_data = copy.deepcopy(data)
        # get the region of car instance 26002
            # print(data)
            feat_map = sample_features(data['inst'])
            generated1 = model.inference(data['label'], data['inst'], data['image'], feat_map=feat_map)
            # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            #                     ('synthesized_image', util.tensor2im(generated1.data[0]))])
            img_path = data['path']
            print('process image... %s' % img_path)
            # visualizer.save_images(webpage, visuals, img_path)

            # bicycle = (data['inst'] == 33000).nonzero().numpy()
            # bicycle[:, 3] -= 300
            # np.save('bicycle.npy', bicycle)
            # rider = (data['inst'] == 25000).nonzero().numpy()
            # rider[:, 3] -= 300
            # np.save('rider.npy', rider)
            # np_region = car_region.numpy()
            # np.save('person.npy', np_region)
            new_data = {}
            new_data['inst'] = data['inst'].clone()
            new_data['label'] = data['label'].clone()
            new_data['image'] = data['image'].clone()
            # feat_map, new_data = remove_object(feat_map, new_data, car_region)
            if opt.object == 'bicycle':
                could_add, feat_map, new_data = add_object(feat_map, new_data, "../generators/pix2pixHD-master/bicycle.npy", 33)
                if could_add:
                    could_add, feat_map, new_data = add_object(feat_map, new_data, "../generators/pix2pixHD-master/rider.npy", 25)
                else:
                    continue

            elif opt.object == 'vehicle':
                could_add, feat_map, new_data = add_object(feat_map, new_data, "../generators/pix2pixHD-master/car.npy", 26)

            elif opt.object == 'pedestrian':
                could_add, feat_map, new_data = add_object(feat_map, new_data, "../generators/pix2pixHD-master/person.npy", 24)
            
            if not could_add:
                continue

            # could_add, feat_map, new_data = add_object(feat_map, new_data, 'car_vert_same_direction_close_distance.npy')
            # if not could_add:
            #     continue
            generated2 = model.inference(new_data['label'], new_data['inst'], new_data['image'], feat_map=feat_map)
            
            # save_image(generated1, source_data_path, str(i) + '.jpg')
            # save_image(generated2, follow_up_data_path, str(i) + '.jpg')

            save_image(generated1, source_data_path, str(i) + '.png')
            save_image(generated2, follow_up_data_path, str(i) + '.png')

            # visuals = OrderedDict([('input_label', util.tensor2label(new_data['label'][0], opt.label_nc)),
            #                     ('synthesized_image', util.tensor2im(generated2.data[0]))])

            # img_path = [img_path[0][:-4] + 'add_a_car.png']
            # visualizer.save_images(webpage, visuals, img_path)
            
    webpage.save()

#python image_control.py --checkpoints_dir ./checkpoints --name label2city --dataroot ../../source_datasets --output_path ../../follow_up_datasets --feature bicycle.npy --feature_ex rider.npy --add_object bicycle