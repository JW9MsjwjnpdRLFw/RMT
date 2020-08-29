import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 
import os
import numpy as np
from driving_model import *
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号#有中文出现的情况，需要u'内容'

def get_original_label(source_path, follow_path, model):
    image_list = os.listdir(follow_path)
    original_label = [0 for i in range(500)]
    follow_label = [0 for i in range(500)]
    for test_image in image_list:
        img = mpimg.imread(os.path.join(source_path, test_image))
        img = img / 255.
        img = torch.from_numpy(np.transpose(img, (-1, 0, 1))).unsqueeze(0)
        img = img.type(torch.FloatTensor)
        img = img.to(device)
        # img_label = test_set_label[int(test_image[:-4])]
        pred = model(img).item()   
        original_label[int(test_image[:-4])] = pred

        img = mpimg.imread(os.path.join(follow_path, test_image))
        img = img / 255.
        img = torch.from_numpy(np.transpose(img, (-1, 0, 1))).unsqueeze(0)
        img = img.type(torch.FloatTensor)
        img = img.to(device)
        # img_label = test_set_label[int(test_image[:-4])]
        pred = model(img).item()   
        follow_label[int(test_image[:-4])] = pred
    return np.array(original_label), np.array(follow_label)  

def get_label_file():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet101'
    original_model = build_resnet101(False)
    original_model.load_state_dict(torch.load('models/model - resnet101.pt'))
    original_model = original_model.to(device)
    original_model.eval()
    image_pairs = [('night', 'source_test_set/original', 'follow_up_test_set/night', 217),
                    ('rainy','source_test_set/original', 'follow_up_test_set/rainy', 217),
                    ('add_car','source_test_set/gen_car', 'follow_up_test_set/add_car', 147),
                    ('add_bicycle','source_test_set/gen_bicycle', 'follow_up_test_set/add_bicycle', 126),
                    ('add_person','source_test_set/gen_person', 'follow_up_test_set/add_person', 168)
                ]
    for (scenario, source_set_path, test_set_path, _) in image_pairs:
        ori_label, follow_label = get_original_label(source_set_path, test_set_path, original_model)
        f = pd.DataFrame(np.hstack([ori_label.reshape(-1, 1), follow_label.reshape(-1, 1)]), columns=['ori_pred', 'mod_pred'])
        f.to_csv(scenario + '_label_resnet101.csv') 


image_pairs = [('night', 'source_test_set/original', 'follow_up_test_set/night', 217),
                ('rainy','source_test_set/original', 'follow_up_test_set/rainy', 217),
                ('add_car','source_test_set/gen_car', 'follow_up_test_set/add_car', 147),
                ('add_bicycle','source_test_set/gen_bicycle', 'follow_up_test_set/add_bicycle', 126),
                ('add_person','source_test_set/gen_person', 'follow_up_test_set/add_person', 168)
             ]

for (scenario, original, generated, sample_num) in image_pairs:
    image_list = os.listdir(generated)
    sample_list = np.random.choice(image_list, sample_num, replace=False)

    prediction_epoch = pd.read_csv(scenario + '_label_basecnn.csv')
    prediction_vgg16 = pd.read_csv(scenario + '_label_vgg16.csv')
    prediction_resnet101 = pd.read_csv(scenario + '_label_resnet101.csv')


    for image in sample_list:
        original_image = mpimg.imread(os.path.join(original, image))
        generated_image = mpimg.imread(os.path.join(generated, image))
        image_index = int(image[:-4])

        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(original_image)

        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(generated_image)
        plt.text(-200, 280, 'Epoch prediction: %.4f; %.4f' % (prediction_epoch.iloc[image_index]['ori_pred'], prediction_epoch.iloc[image_index]['mod_pred']))
        plt.text(-200, 310, 'VGG16 prediction: %.4f; %.4f' % (prediction_vgg16.iloc[image_index]['ori_pred'], prediction_vgg16.iloc[image_index]['mod_pred']))
        plt.text(-200, 340, 'Resnet101 prediction: %.4f; %.4f' % (prediction_resnet101.iloc[image_index]['ori_pred'], prediction_resnet101.iloc[image_index]['mod_pred']))
        # plt.show()
        plt.savefig('evaluation\\' + scenario + "\\" + image[:-4])
        plt.close('all')