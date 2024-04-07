"""
Split data into 70% train, 10% val, 20% test
and then write to directories that the pytorch
ImageFolder expects.

Writes images to directories of the following structure:

train
    |
    Class directories with images
val
    |
    Class directories with images
test
    |
    Class directories with images
"""

import os
import yaml
import shutil
from sklearn.model_selection import train_test_split
from get_metadata import get_metadata

data_root = "path/to/1769319269-DND-Diko-WWWR/DND-Diko-WWWR"
croptype = 'WW2020' # WW2020, WR2021
split = 'trainval' # trainval, test

img_paths, label_names, class_to_ind = get_metadata(data_root, croptype, split, verbose=True)

dest_root = "path/to/data/destination"

croptypes = ["WW2020","WR2021"]
for croptype in croptypes:
    dir_path = os.path.join(data_root, croptype)

    # load split
    split_path = os.path.join(data_root, croptype, "trainval")+'.txt'
    with open(split_path, 'r') as f:
        file_names = f.read().splitlines()
        print("Loading split from: ".ljust(40), split_path)

    # get labels
    metadata_path = os.path.join(data_root, croptype, 'labels_trainval.yml')
    labels_trainval = yaml.safe_load(open(metadata_path, 'r')) # dict, e.g., {20200422_1.jpg: unfertilized, ...}
    print("Loading labels from: ".ljust(40), metadata_path)

    labels = [labels_trainval[file_name] for file_name in file_names]

    # stratified split into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(file_names, labels, stratify=labels, test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, shuffle=True)

    # make train, val, test folders with relevant images
    for idx, file_name in enumerate(X_train):
        img_path = os.path.join(data_root, croptype, "images", file_name)
        label = y_train[idx]

        dest_path = os.path.join(dest_root, croptype, "train", label)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy2(img_path, dest_path)

    
    for idx, file_name in enumerate(X_val):
        img_path = os.path.join(data_root, croptype, "images", file_name)
        label = y_val[idx]

        dest_path = os.path.join(dest_root, croptype, "val", label)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy2(img_path, dest_path)

    for idx, file_name in enumerate(X_test):
        img_path = os.path.join(data_root, croptype, "images", file_name)
        label = y_test[idx]

        dest_path = os.path.join(dest_root, croptype, "test", label)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy2(img_path, dest_path)