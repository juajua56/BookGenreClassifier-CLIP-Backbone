import os
import pandas as pd

def get_absolute_path(row, data_root):
    folder_name = row['label'].split('.')[0]
    folder_path = os.path.join(data_root, folder_name)
    image_path = os.path.join(folder_path, row['Filename'])
    return image_path

def add_absolute_paths(data, data_root):
    data['Absolute_Path'] = data.apply(lambda row: get_absolute_path(row, data_root), axis=1)
    return data

def add_label_index(data):
    unique_labels = data['label'].unique()
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    data['label_index'] = data['label'].map(label_mapping)
    return data

def freeze_parameters(model, frozen_params):
    for name, param in model.named_parameters():
        if name not in frozen_params:
            param.requires_grad_(False)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} is trainable.")
        else:
            print(f"Parameter {name} is frozen.")
