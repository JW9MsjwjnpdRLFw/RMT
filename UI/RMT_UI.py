import PySimpleGUI as sg
from PIL import Image
# from config import *
import os
import subprocess
import cv2
from xml_parser import updateGenerator, GeneratorObject, addGenerator, parse_new, TransformationObject, ParamObject
import json
import sys
sys.path.append("..")
from data_a2d2 import A2D2Diff, A2D2MT
import importlib
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import copy
import matplotlib.image as mpimg
import numpy as np

THRESHOLD_LOW = 0.1
THRESHOLD_HIGH = 0.2

def get_param_layout():
    param_layout = []
    for param_info in trans_params_all:
        if len(param_info.keys()) > 1:
            param_layout.append([sg.Text("Parameters for " + param_info['trans'], size=(30, 1))])
            for key in param_info.keys():
                if key != 'trans' and 'type' not in key:
                    param_values = list(set(param_info[key]))
                    param_types = list(set(param_info[key+'_type']))
                    if len(param_types) == 1 and param_types[0] == "fixed":
                        param_layout.append([sg.Text(key+":", size=(20, 1)), \
                        sg.Combo(param_values, size=(10, 1),disabled=True, default_value=param_values[0], key=param_info['trans'] + '_' + key)])
                    else:
                        param_layout.append([sg.Text(key+":", size=(20, 1)), \
                        sg.Input(size=(20, 1),disabled=True, key=param_info['trans'] + '_' + key)])
            param_layout.append([sg.Text('_'*50)])                       
    return param_layout

def get_input_layout():
    transformation_layout = [
        [sg.Text("Transformations", size=(15, 1))], 
        # [sg.Checkbox("Inject", key="Inject", enable_events=True, size=(14, 1)), \
        # sg.Checkbox("ChangeWeather", key="ChangeWeather", enable_events=True, size=(14, 1)), \
        #  sg.Checkbox("RemoveLine", key="RemoveLine", enable_events=True, size=(14, 1))],
        [sg.Radio(trans, "transformations", key=trans, enable_events=True, size=(14, 1)) for trans in transformations],
        [sg.Text("Transformation engine", size=(30, 1))],
        # [sg.Radio("OpenCV", "transform_engine", key="OpenCV"), sg.Radio("Pix2pixHD", "transform_engine", key="Pix2pixHD"), \
        #     sg.Radio("UNIT", "transform_engine", key="UNIT"), sg.Radio("SCNN", "transform_engine", key="SCNN")],
        [sg.Radio(engine, "transform_engine", disabled=True,enable_events=True, key=engine) for engine in engines],
        [sg.Text('_'*50)],
    ]

    param_layout = get_param_layout()
    transformation_layout = transformation_layout + param_layout
    transformation_layout.append([sg.Text("Apply transformation for"),
     sg.Combo(values=["X_N1", "X_N2"], key="trans_subject", size=(14,1), default_value="X_N1"),
     sg.Button("Add the transformation")])
    input_layout = [
        [sg.Frame(layout = transformation_layout,

            title='Transformation setting', relief=sg.RELIEF_SUNKEN)],
        
        [sg.Frame(layout=[
            [sg.Text("Transformations for X_N1: None", size=(45, 2), key="Trans1", text_color="red"), sg.Button("Remove Transformation", key="Remove1", visible=False)],
            [sg.Text("Transformations for X_N2: None", size=(45, 2),key="Trans2", text_color="red"), sg.Button("Remove Transformation", key="Remove2", visible=False)],
            ],
            title='Transformations to apply', relief=sg.RELIEF_SUNKEN)],            
        

        [sg.Frame(layout=[
            # [sg.Text("Input pair (X_N1, X_N2):", size=(20, 1)), sg.Combo(("(Original image, Transformed image)", "(Transformed images 1 and 2)"), size=(30, 1), default_value="(Original image, Transformed image)", key="MRV_pair_type")],
            [sg.Text("Input dataset:", size=(20, 1)), sg.Combo(datasets, size=(30, 1), default_value=datasets[0], key="MRV_pair_data")],
            [sg.Text("Model:", size=(20, 1)), sg.Combo(models, size=(30, 1), default_value=models[0], key="MRV_MUT")],
            # [sg.Text("Inequation:", size=(20, 1)), sg.Combo(("decreaseRatio", "current", "deviation"), size=(20, 1), default_value="decreaseRatio", key="MRV_RF")],
            [sg.Text("Higher order function:", size=(20, 1)), sg.Input(key="MRV_RF", size=(30, 1), )],
            # [sg.Text("Range:", size=(20, 1)), sg.Input(key="MRV_Range_low", size=(11, 1)), sg.Text("to", size=(5,1)), sg.Input(key="MRV_range_high", size=(11, 1))],
            ],
            title='MR setting', relief=sg.RELIEF_SUNKEN)],

        [sg.Button("Generate test"), sg.Button("Configs"), sg.Cancel()]  
    ]
    return input_layout

def get_info(transformation_list, model_list, data_list):
    transformations = []
    engines = []
    trans_params = [] # for each transformation with different parameter
    trans_params_all = []
    models = []
    datasets = []
    for trans in transformation_list:
        transformations.append(trans.name)
        engines.append(trans.engine)
        if len(trans.params) != 0:
            for param in trans.params:
                trans_params.append((trans.name, trans.engine, param.name, param.type, param.value))
        else:
            trans_params.append((trans.name, trans.engine, None, None, None))

    transformations = list(set(transformations))
    transformations.sort()

    for trans_name in transformations:
        param_dict = {'trans': trans_name}
        for trans in transformation_list:
            if trans.name == trans_name:
                for param in trans.params:
                    if param.name not in param_dict.keys():
                        param_dict[param.name] = [param.value]
                        param_dict[param.name + '_type'] = [param.type]
                    else:
                        param_dict[param.name].append(param.value)
                        param_dict[param.name + '_type'].append(param.type)
        trans_params_all.append(param_dict)

    engines = list(set(engines))
    engines.sort()

    for model in model_list:
        models.append(model.name)
    
    for dataset in data_list:
        datasets.append(dataset.name)

    return transformations, engines, trans_params, trans_params_all, models, datasets

def update_param_status():
    temp_trans_params = {}
    # disable parameter settings of unselected transformations
    for trans in trans_params_all:
        if trans['trans'] != current_trans.name:
            for param_key in trans.keys():
                if param_key != 'trans' and 'type' not in param_key:
                    input_window[trans['trans'] + '_' + param_key].update(disabled=True)
    
    for trans in transformation_list:
        if trans.name == current_trans.name and trans.engine == current_trans.engine:
            for param in trans.params:
                input_window[current_trans.name + '_' + param.name].update(param.value)
            break

    # activate parameter settings of selected transformation
    for trans in trans_params:
        if trans[0] == current_trans.name and trans[1] == current_trans.engine:
            if trans[2] != None and trans[2] not in temp_trans_params.keys():
                temp_trans_params[trans[2]] = [trans[4]]
                temp_trans_params[trans[2] + '_type'] = [trans[3]]
            elif trans[2] in temp_trans_params.keys():
                temp_trans_params[trans[2]].append(trans[4])
                temp_trans_params[trans[2] + '_type'].append(trans[3])
    
    for param_key in temp_trans_params.keys():
        if param_key != 'trans' and 'type' not in param_key:
            param_values = list(set(temp_trans_params[param_key]))
            param_types = list(set(temp_trans_params[param_key + '_type']))
            # if len(param_types) == 1 and param_types[0] == "fixed":
            if 'Combo' in str(type(input_window[current_trans.name + '_' + param_key])):
                input_window[current_trans.name + '_' + param_key].update(value=param_values[0], \
                    values=param_values, disabled=False)
            elif len(param_types) == 1 and param_types[0] == "fixed":
                input_window[current_trans.name + '_' + param_key].update(value=param_values[0])
            else:
                input_window[current_trans.name + '_' + param_key].update(value=param_values[0], disabled=False)

def reset_transformation_setting():
    for trans in transformations:
        input_window[trans].update(value=False)
    
    for engine in engines:
        input_window[engine].update(value=False, disabled=True)

    for trans in trans_params_all:
        for param_key in trans.keys():
            if param_key != 'trans' and 'type' not in param_key:
                input_window[trans['trans'] + '_' + param_key].update(disabled=True)


def make_predictions_a2d2(model, x_n1, x_n2):
    with torch.no_grad():
        bg_speed = []
        label = []
        source_pred = []
        follow_up_pred = []
        for i in range(len(x_n1)):
            source_images = x_n1[i][0]
            source_bg_speed = x_n1[i][1]
            source_label = x_n1[i][2]

            follow_up_images = x_n2[i][0]
            follow_up_bg_speed = x_n2[i][1]
            follow_up_label = x_n2[i][2]
            label.append(source_label)
            source_images = source_images.type(torch.FloatTensor)
            follow_up_images = follow_up_images.type(torch.FloatTensor)
            source_speed = torch.tensor(source_bg_speed).type(torch.FloatTensor)
            follow_speed = torch.tensor(follow_up_bg_speed).type(torch.FloatTensor)
            # print(source_speed, follow_speed)
            source_input = (source_images.unsqueeze(0).to(device), source_speed.unsqueeze(0).to(device))
            follow_up_input = (follow_up_images.unsqueeze(0).to(device), follow_speed.unsqueeze(0).to(device))
            source_output = model(source_input)
            follow_up_output = model(follow_up_input)
            bg_speed.append(source_speed.mean().item())
            source_pred.append(source_output.item())
            follow_up_pred.append(follow_up_output.item())
        return  source_pred, follow_up_pred


def resize_img(dataset, x_n):
    if dataset.name == "A2D2":
        for d in os.listdir(dataset.path):
            if 'png' in d:
                img = cv2.imread(os.path.join(dataset.path, d))
                img = img[161:1208, 442:1489]
                resize_img = cv2.resize(img, (int(dataset.img_size), int(dataset.img_size)))
                image_json = d[:-4] + '.json'
                with open(os.path.join(dataset.path,image_json), 'r') as f:
                    image_info = json.load(f)            
                    timestamp = image_info["cam_tstamp"]
                    # cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)
                    cv2.imwrite(os.path.join("../test_images/" + x_n, str(timestamp) + '.png'), resize_img)
    elif dataset.name == "Cityscapes":
        for d in os.listdir(dataset.path):
            if 'png' in d:
                img = cv2.imread(os.path.join(dataset.path, d))
                img = img[0:852, 384:1520]
                resize_img = cv2.resize(img, (int(dataset.img_size), int(dataset.img_size)))
                cv2.imwrite(os.path.join("../test_images/" + x_n, d), resize_img)


def create_mt_set(dataset, transformations, x_n):
    if len(transformations) == 0:
        resize_img(dataset, x_n)
    elif len(transformations) == 1:
        script = transformations[0].running_script
        for param in transformations[0].params:
            if param.check == "1":
                script += " --%s %s" % (param.name, param.value)
        if transformations[0].name == "ChangeScene":
            resize_img(dataset, x_n)
            script += " --dataset_path %s --output_path %s" % ("../test_images/" + x_n, "../test_images/" + x_n)
            print(script)
            os.system(script)
        else:
            script += " --dataset_path %s --output_path %s" % (dataset.path, "../test_images/" + x_n)
            print(script)
            os.system(script)
    else:
        for i, trans in enumerate(transformations):
            script = trans.running_script
            for param in trans.params:
                if param.check == "1":
                    script += " --%s %s" % (param.name, param.value)
            if i == 0:
                script += " --dataset_path %s --output_path %s" % (dataset.path, "../test_images/" + x_n)
            else:
                script += " --dataset_path %s --output_path %s" % ("../test_images/" + x_n, "../test_images/" + x_n)
            print(script)
            os.system(script)
    # else:
    #     resize_img(dataset, x_n)
def get_output_layout(mt_result):
    img_list = os.listdir("../test_images/x_n1")[:3]

    for i, img_name in enumerate(img_list):
        Image.open(os.path.join("../test_images/x_n1", img_name)).save("source_{}.png".format(i))
        Image.open(os.path.join("../test_images/x_n2", img_name)).save("follow_up_{}.png".format(i))

    out_layout = [      
        [sg.Text(mt_result)],
            [sg.Frame(layout = [
                # [sg.Image('original_1.png'),
                #  sg.Image('original_2.png'),
                #  sg.Image('original_3.png')]
                [sg.Image("source_{}.png".format(i)) for i in range(3)]
                 ], 
                title  ='Original graph', relief =sg.RELIEF_SUNKEN)],
            [sg.Frame(layout = [
                # [sg.Image('result_1.png'),
                #  sg.Image('result_2.png'),
                #  sg.Image('result_3.png')]
                [sg.Image("follow_up_{}.png".format(i)) for i in range(3)]

                 ], 
                title = 'Generated graph', relief =sg.RELIEF_SUNKEN)],
        # [sg.Image('../source_datasets/orginal/1.jpg'), sg.Image('../follow_up_datasets/rainy/2.jpg')],      
        [sg.OK(key="result_ok")]
    ]
    return out_layout
def clear_test_images():
    img_list = os.listdir("../test_images/x_n1")
    for img_name in img_list:
        os.remove(os.path.join("../test_images/x_n1", img_name))
    img_list = os.listdir("../test_images/x_n2")
    for img_name in img_list:
        os.remove(os.path.join("../test_images/x_n2", img_name))

if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    transformation_list, model_list, data_list = parse_new()
    transformations, engines, trans_params, trans_params_all, models, datasets = get_info(transformation_list, model_list, data_list)
    selected_trans = []
    trans_for_x1 = []
    trans_for_x2 = []
    input_window = sg.Window('Rule-based Metamorphic Testing').Layout(get_input_layout()) 
    current_trans = TransformationObject([None, None, None, None])

    while True:
        button, values = input_window.Read()
        print(button, values)
        if button in (None, 'Exit'):
            break
        
        elif button in transformations:
                current_trans.name = button
                support_engines = []
                for trans in trans_params:
                    if button == trans[0] and trans[1] not in support_engines:
                        support_engines.append(trans[1])
                
                for engine in engines:
                    if engine in support_engines:
                        input_window[engine].update(disabled=False)
                    else:
                        input_window[engine].update(disabled=True)
                
                input_window[support_engines[0]].update(value=True)
                current_trans.engine = support_engines[0]
                update_param_status()

        elif button in engines:
            # print(button)
            current_trans.engine = button
            update_param_status()

        elif button == "Add the transformation":
            if current_trans.name and current_trans.engine:
                for transformation in transformation_list:
                    if transformation.name == current_trans.name and transformation.engine == current_trans.engine:
                        
                        current_trans.params = copy.deepcopy(transformation.params)
                        current_trans.running_script = transformation.running_script
                        break
                for param in current_trans.params:
                    param.value = values[current_trans.name + '_' + param.name]

                # params_for_trans = {}
                # for trans in trans_params_all:
                #     if trans['trans'] == current_trans.name:
                #         for param_key in trans.keys():
                #             if param_key != 'trans' and 'type' not in param_key:
                #                 params_for_trans[param_key] = values[trans['trans'] + '_' + param_key]
                #         break
                
                # current_trans.params = params_for_trans
                # get_running_script()
                # current_trans.running_script = get_running_script()
                # print(current_trans.running_script)
                current_param_str = {}
                for param in current_trans.params:
                    current_param_str[param.name] = param.value
                if values["trans_subject"] == "X_N1":
                    trans_for_x1.append(current_trans)
                    str_trans1 = input_window["Trans1"].DisplayText
                    if "None" in str_trans1:
                        str_trans1 = str_trans1.replace("None", "")

                    str_trans1 += "(%s, %s, %s);" % (current_trans.name, current_trans.engine, current_param_str)
                    input_window["Trans1"].update(str_trans1)
                    input_window["Remove1"].update(visible=True)
                    input_window["Trans1"].SetTooltip(str_trans1)

                elif values["trans_subject"] == "X_N2":
                    trans_for_x2.append(current_trans)
                    str_trans2 = input_window["Trans2"].DisplayText
                    if "None" in str_trans2:
                        str_trans2 = str_trans2.replace("None", "")
                    str_trans2 += "(%s, %s, %s);" % (current_trans.name, current_trans.engine, current_param_str)
                    input_window["Trans2"].update(str_trans2)  
                    input_window["Remove2"].update(visible=True)
                    input_window["Trans2"].SetTooltip(str_trans2)

                # if len(selected_trans) == 0:
                #     input_window['Trans1'].update("%s, %s, %s" % \
                #          (current_trans.name, current_trans.engine, current_trans.params))
                # else:
                #     input_window['Trans2'].update("%s, %s, %s" % \
                #          (current_trans.name, current_trans.engine, current_trans.params))                        
                
                # selected_trans.append(current_trans)
                current_trans = TransformationObject([None, None, None, None])
                reset_transformation_setting()
        elif button == "Remove1":
            trans_for_x1 = []
            input_window["Remove1"].update(visible=False)
            input_window["Trans1"].update("Transformations for X_N1: None")
        elif button == "Remove2":
            trans_for_x2 = []
            input_window["Remove2"].update(visible=False)
            input_window["Trans2"].update("Transformations for X_N2: None")
        elif button == "Generate test":
            clear_test_images()
            # read MRV settings
            # test_mode = values["MRV_pair_type"]
            selected_dataset = None
            for dataset in data_list:
                if dataset.name == values["MRV_pair_data"]:
                    selected_dataset = dataset
                    break
            selected_model = None
            for model in model_list:
                if model.name == values["MRV_MUT"]:
                    selected_model = model
            
            # oracle_range = [values["MRV_Range_low"], values["MRV_Range_high"]]

            # create metamorphic test set X_N1 and X_N2
            create_mt_set(selected_dataset, trans_for_x1, "x_n1")
            create_mt_set(selected_dataset, trans_for_x2, "x_n2")

            # load driving models 
            driving_model_module = __import__(selected_model.class_file)
            driving_model =  getattr(driving_model_module, selected_model.class_name)()
            if selected_model.distributed == "1":
                driving_model = nn.DataParallel(driving_model, device_ids=[0, 1])
            
            driving_model.load_state_dict(torch.load(selected_model.path))
            driving_model = driving_model.to(device)
            driving_model.eval()

            # make predictions for mt sets
            if selected_dataset.name == "A2D2":
                split_path = selected_dataset.path.split(os.sep)
                root_path = split_path[0]
                for i in range(1, split_path.index('a2d2')+1):
                    root_path += os.sep
                    root_path += split_path[i]
                # root_path = os.path.join(*split_path[:split_path.index('a2d2')+1])
                mt_set_x1 = A2D2MT(root_path, mt_path="../test_images/x_n1", mode="self")
                mt_set_x2 = A2D2MT(root_path, mt_path="../test_images/x_n2", mode="self")
                pred_x1, pred_x2 = make_predictions_a2d2(driving_model, mt_set_x1, mt_set_x2)
            elif selected_dataset.name == "Cityscapes":
                pred_x1 = []
                pred_x2 = []
                for img_name in os.listdir("../test_images/x_n2"):
                    try:
                        x_1 = mpimg.imread(os.path.join("../test_images/x_n1", img_name))
                        x_2 = mpimg.imread(os.path.join("../test_images/x_n2", img_name))
                    except:
                        continue
                    if np.max(x_1) > 1:
                        x_1 = x_1 / 255.
                    img_tensor_1 = torch.from_numpy(np.transpose(x_1, (-1, 0, 1))).unsqueeze(0)
                    img_tensor_1 = img_tensor_1.type(torch.FloatTensor)
                    img_tensor_1 = img_tensor_1.to(device)
                    if np.max(x_2) > 1:
                        x_2 = x_2 / 255.
                    img_tensor_2 = torch.from_numpy(np.transpose(x_2, (-1, 0, 1))).unsqueeze(0)
                    img_tensor_2 = img_tensor_2.type(torch.FloatTensor)
                    img_tensor_2 = img_tensor_2.to(device)
                    pred_x1.append(driving_model(img_tensor_1).item())
                    pred_x2.append(driving_model(img_tensor_2).item())

            # create metamorphic relation
            relation_function = lambda X_N1, X_N2: eval(values["MRV_RF"])
            total = len(pred_x1)
            violation = 0
            for (y1, y2) in zip(pred_x1, pred_x2):
                if not relation_function(y1, y2):
                    violation += 1
            print(violation, total)
            # display result window
            mt_result = "{} out of {} violations were founded.".format(violation, total)
            selected_dataset = None
            selected_model = None
            trans_for_x1 = []
            trans_for_x2 = []
            input_window.close()
            out_window = sg.Window('show').Layout(get_output_layout(mt_result))   
            button, values = out_window.Read()
            if button == "result_ok":
                out_window.close()
                for i in range(3):
                    os.remove("source_{}.png".format(i))
                    os.remove("follow_up_{}.png".format(i))
                input_window = sg.Window('Rule-based Metamorphic Testing').Layout(get_input_layout())              

            # if test_mode == "(OTC, MTC)":
            #     if len(selected_trans) == 1:
            #         input_path = selected_dataset.path
            #         output_path = "../MTC/MTC"
            #         script = selected_trans[0].running_script  + " --input_path {} --output_path {}".format(input_path, output_path)
            #         print(script)
            #         os.system(script)
            #         model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "source_datasets"))
            #         model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "follow_up_datasets"))


            #     elif len(selected_trans) == 2:
            #         add_trans = None
            #         change_trans = None

            #         if selected_trans[0].name == "Inject":
            #             add_trans = selected_trans[0]
            #             change_trans = selected_trans[1]
            #         else:
            #             add_trans = selected_trans[1]
            #             change_trans = selected_trans[0]
                    
            #         input_Path = selected_dataset.path
            #         output_path = "../MTC/MTC"
            #         temp_output_path = "../MTC/temp"

            #         script1 = add_trans.running_script + " --input_path {} --output_path {}".format(input_path, temp_output_path)
            #         script2 = change_trans.running_script + " --input_path {} --output_path {}".format(os.path.join(temp_output_path, "follow_up_datasets"), output_path)
            #         os.system(script1)
            #         os.system(script2)

            #         model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "source_datasets"))
            #         model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "follow_up_datasets"))

            # elif test_mode == "(MTC1, MTC2)" and len(selected_trans) == 2:
            #     input_path = selected_dataset.path
            #     output_path1 = "../MTC/MTC1"
            #     output_path2 = "../MTC/MTC2"

            #     script1 = selected_trans[0].running_script  + " --input_path {} --output_path {}".format(input_path, output_path1)
            #     script2 = selected_trans[1].running_script  + " --input_path {} --output_path {}".format(input_path, output_path2)
            #     os.system(script1)
            #     os.system(script2)

            #     model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path1, "follow_up_datasets"))
            #     model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path2, "follow_up_datasets"))


            # result_source = subprocess.check_output(model_script1)
            # result_source = str(result_source).split("\\n")[1][1:-1].split("]")[0].split(", ") 
            # result_follow_up = subprocess.check_output(model_script2)  
            # result_follow_up = str(result_follow_up).split("\\n")[1][1:-1].split("]")[0].split(", ") 

            # violation, total = mt_check(result_source, result_follow_up, relation_function, oracle_range)