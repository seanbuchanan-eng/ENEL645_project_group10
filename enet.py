"""
Training script for the EfficientNet_v2_m model.
"""

#! /usr/bin/env python 
# ---------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import modelling_utils
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
YEAR = "All" # WW2020 or WR2021 or All
run_env = "talc"
if run_env == "local":
    data_dir = "C:/Users/seanb/ENEL645/project_data/" + YEAR
    num_workers = 0
    weights_path = "weights/Res_50.pth"
    param_file_name = "best_params.pth"
    result_file = "results.npy"
    batch_size = 32
    param_path = "weights/"
    epochs = 1
    learn_rate = 1e-3
elif run_env == "talc":
    data_dir = "/work/TALC/enel645_2024w/sean-chris-chris/" + YEAR
    num_workers = 4
    weights_path = "/work/TALC/enel645_2024w/sean-chris-chris/weights/Res_50.pth"
    param_file_name = "enet_params.pth"
    result_file = "enet_results.npy"
    batch_size = 8
    param_path = ""
    epochs = 100
    learn_rate = 1e-5
else:
    print("Wrong run environment chosen! exiting.")
    exit()
# ----------------------------------------------------------------------

# import data
img_transforms = modelling_utils.enet_transforms

img_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), img_transforms[x]) for x in ["train", "val", "test"]
}

data_loaders = {
    x: torch.utils.data.DataLoader(img_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ["train", "val", "test"]
}

dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val', 'test']}
class_names = img_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create model
model = modelling_utils.Enet()

model = model.to(device)
print("model loaded successfully")

# select loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learn_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

model = modelling_utils.train_validate(model, criterion, optimizer, param_path, param_file_name, 
                                       data_loaders, device, dataset_sizes, result_file,
                                       scheduler=scheduler, num_epochs=epochs)

modelling_utils.test(model, data_loaders["test"], device)
modelling_utils.calc_metrics(model, data_loaders["test"], device)

