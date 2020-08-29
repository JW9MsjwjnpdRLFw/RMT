import os
import pandas as pd 
import numpy as np

root = 'D:\\MT\\mt_framework\\evaluation'
folder = ['add_car', 'add_bicycle', 'add_person', 'night', 'rainy']
model = ['basecnn', 'vgg16', 'resnet101']

for f in folder:
    img_list = os.listdir(os.path.join(root, f))
    for m in model:
        data = [[0, 0] for i in range(500)]
        df = pd.read_csv(f + '_label_' + m + '.csv')
        for img in img_list:
            index = int(img[:-4])
            ori_pred = df.iloc[index]['ori_pred']
            mod_pred = df.iloc[index]['mod_pred']
            data[index] = [ori_pred, mod_pred]
    
        data = np.array(data)
        df = pd.DataFrame(data, columns=['ori_pred', 'mod_pred'])
        df.to_csv('filter_data\\' + f + '_' + m + '.csv')

