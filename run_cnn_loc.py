#!/usr/bin/python

from __future__ import print_function
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


src_path = sys.argv[1]
check_path = sys.argv[2]
files_path = sys.argv[3]
results_path = sys.argv[4]
csv_file = sys.argv[5]
res = int(sys.argv[6])
cl_num = int(sys.argv[7])


#print(src_path)
#print(check_path)
#print(files_path)
#print(results_path)
#print(csv_file)
#print(res)
#print(cl_num)

# TR17 version checks

import torch
import torchvision
import pycocotools

print("torch version = ", torch.__version__)
print("torchvision version = ", torchvision.__version__)
print("torch CUDA version = ", torch.version.cuda)
print("torch cuDNN version = ", torch.backends.cudnn.version())
print("torch cuDNN is_available = ", torch.backends.cudnn.is_available())
print("")

os.chdir(src_path)




from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md



os.chdir(files_path)


n_epochs = 300
all_cl_num = cl_num + 1

args = ArgumentsTrainTestLocalisation(epochs=n_epochs,
                                      batch_size=2,
                                      lr=0.002,
                                      crop_height=res,
                                      crop_width=res,
                                      crop_depth=res,
                                      validation_steps=8,
                                      lamda=10,
                                      training=False,
                                      testing=False,
                                      running=True,
                                      root_dir=files_path,
                                      csv_dir=files_path,
                                      train_csv=csv_file,
                                      valid_csv=csv_file,
                                      test_csv=csv_file,
                                      run_csv=csv_file,
                                      results_dir=results_path,
                                      checkpoint_dir=check_path,
                                      exp_name='Loc_3D',
                                      task_net='unet_3D',
                                      n_classes=all_cl_num)



args.gpu_ids = [0]

model = md.LocalisationNetwork3DMultipleLabels(args)
model.run(args,0)





