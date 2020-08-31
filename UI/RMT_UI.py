import PySimpleGUI as sg
from PIL import Image
# from config import *
import os
import subprocess
import cv2
from xml_parser import updateGenerator, GeneratorObject, addGenerator, parse_new, TransformationObject, ParamObject


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

def get_running_script():
    # current_params = list(current_trans.params.keys())
    # current_params.sort()

    for trans in transformation_list:
        if trans.name == current_trans.name and trans.engine == current_trans.engine:
            print(1)
            if len(current_trans.params) == 0 and len(trans.params) == 0:
                running_script = trans.running_script
                break

            trans_params = {}
            for param in trans.params:
                if param.check == "1":
                    trans_params[param.name] = param.value

            match = True
            for key, value in trans_params.items():
                if key not in current_trans.params.keys() or current_trans.params[key] != value:
                    match = False
                    break
            
            if match:
                running_script = trans.running_script

    return running_script

def mt_check(result_source, result_follow_up, relation_funtion, oracle_range):
    violation = 0
    for prediction_source, prediction_follow in zip(result_source, result_follow_up):
        prediction_source = float(prediction_source)
        prediction_follow = float(prediction_follow)

        if relation_function == "decreaseRatio":
            if prediction_follow < prediction_source * (1 - oracle_range[1]) \
                 or prediction_follow > prediction_source * (1 - oracle_range[0]):
                violation += 1
        elif relation_function == "current":
            if prediction_follow < oracle_range[0] or prediction_follow > oracle_range[1]:
                violation += 1
        elif relation_function == "deviation":
            if prediction_source - prediction_follow < oracle_range[0] or prediction_source - prediction_follow > oracle_range[1]:
                violation += 1
        # if prediction_follow < prediction_source * (1 - threshold_high) or prediction_follow > prediction_source * (1 - threshold_low):
        #         violation += 1
    return violation, len(result_source)

if __name__ == "__main__":
    transformation_list, model_list, data_list = parse_new()
    transformations, engines, trans_params, trans_params_all, models, datasets = get_info(transformation_list, model_list, data_list)
    selected_trans = []
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
                params_for_trans = {}
                for trans in trans_params_all:
                    if trans['trans'] == current_trans.name:
                        for param_key in trans.keys():
                            if param_key != 'trans' and 'type' not in param_key:
                                params_for_trans[param_key] = values[trans['trans'] + '_' + param_key]
                        break
                
                current_trans.params = params_for_trans
                # get_running_script()
                current_trans.running_script = get_running_script()
                print(current_trans.running_script)

                if values["trans_subject"] == "X_N1":
                    str_trans1 = input_window["Trans1"].DisplayText
                    if "None" in str_trans1:
                        str_trans1 = str_trans1.replace("None", "")
                    str_trans1 += "(%s, %s, %s);" % (current_trans.name, current_trans.engine, current_trans.params)
                    input_window["Trans1"].update(str_trans1)
                    input_window["Remove1"].update(visible=True)
                    input_window["Trans1"].SetTooltip(str_trans1)

                elif values["trans_subject"] == "X_N2":
                    str_trans2 = input_window["Trans2"].DisplayText
                    if "None" in str_trans2:
                        str_trans2 = str_trans2.replace("None", "")
                    str_trans2 += "(%s, %s, %s);" % (current_trans.name, current_trans.engine, current_trans.params)
                    input_window["Trans2"].update(str_trans2)  
                    input_window["Remove2"].update(visible=True)
                    input_window["Trans2"].SetTooltip(str_trans2)

                # if len(selected_trans) == 0:
                #     input_window['Trans1'].update("%s, %s, %s" % \
                #          (current_trans.name, current_trans.engine, current_trans.params))
                # else:
                #     input_window['Trans2'].update("%s, %s, %s" % \
                #          (current_trans.name, current_trans.engine, current_trans.params))                        
                
                selected_trans.append(current_trans)
                current_trans = TransformationObject([None, None, None, None])
                reset_transformation_setting()
        
        elif button == "Generate test":
            # read MRV settings
            test_mode = values["MRV_pair_type"]
            selected_dataset = None
            for dataset in data_list:
                if dataset.name == values["MRV_pair_data"]:
                    selected_dataset = dataset
                    break
            selected_model = None
            for model in model_list:
                if model.name == values["MRV_MUT"]:
                    selected_model = model
            
            relation_function = values["MRV_RF"]
            oracle_range = [values["MRV_Range_low"], values["MRV_Range_high"]]

            # create metamorphic test set and make model predictions
            if test_mode == "(OTC, MTC)":
                if len(selected_trans) == 1:
                    input_path = selected_dataset.path
                    output_path = "../MTC/MTC"
                    script = selected_trans[0].running_script  + " --input_path {} --output_path {}".format(input_path, output_path)
                    print(script)
                    os.system(script)
                    model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "source_datasets"))
                    model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "follow_up_datasets"))


                elif len(selected_trans) == 2:
                    add_trans = None
                    change_trans = None

                    if selected_trans[0].name == "Inject":
                        add_trans = selected_trans[0]
                        change_trans = selected_trans[1]
                    else:
                        add_trans = selected_trans[1]
                        change_trans = selected_trans[0]
                    
                    input_Path = selected_dataset.path
                    output_path = "../MTC/MTC"
                    temp_output_path = "../MTC/temp"

                    script1 = add_trans.running_script + " --input_path {} --output_path {}".format(input_path, temp_output_path)
                    script2 = change_trans.running_script + " --input_path {} --output_path {}".format(os.path.join(temp_output_path, "follow_up_datasets"), output_path)
                    os.system(script1)
                    os.system(script2)

                    model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "source_datasets"))
                    model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path, "follow_up_datasets"))

            elif test_mode == "(MTC1, MTC2)" and len(selected_trans) == 2:
                input_path = selected_dataset.path
                output_path1 = "../MTC/MTC1"
                output_path2 = "../MTC/MTC2"

                script1 = selected_trans[0].running_script  + " --input_path {} --output_path {}".format(input_path, output_path1)
                script2 = selected_trans[1].running_script  + " --input_path {} --output_path {}".format(input_path, output_path2)
                os.system(script1)
                os.system(script2)

                model_script1 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path1, "follow_up_datasets"))
                model_script2 = selected_model.running_script + " --model_name {} --model_path {} --input_path {}".format(selected_model.class_name, selected_model.path, os.path.join(output_path2, "follow_up_datasets"))


            result_source = subprocess.check_output(model_script1)
            result_source = str(result_source).split("\\n")[1][1:-1].split("]")[0].split(", ") 
            result_follow_up = subprocess.check_output(model_script2)  
            result_follow_up = str(result_follow_up).split("\\n")[1][1:-1].split("]")[0].split(", ") 

            violation, total = mt_check(result_source, result_follow_up, relation_function, oracle_range)