
# SVRTK : SVR reconstruction based on MIRTK and CNN-based processing for fetal MRI
#
# Copyright 2018-2020 King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# see the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md


# ==================================================================================================================== #
#
#  TRAIN Localisation Network with 3D images
#
# ==================================================================================================================== #

N_epochs = 100
I_size = 128
N_classes = 4


# # # Prepare arguments
args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                      batch_size=2,
                                      lr=0.002,
                                      crop_height=I_size,
                                      crop_width=I_size,
                                      crop_depth=I_size,
                                      validation_steps=8,
                                      lamda=10,
                                      training=True,
                                      testing=False,
                                      running=False,
                                      root_dir='/data/project/Localisation/wshop_data/',
                                      csv_dir='/data/project/Localisation/wshop_data/',
                                      train_csv='data_localisation_3labels_uterus_train.csv',
                                      valid_csv='data_localisation_3labels_uterus_valid.csv',
                                      test_csv='data_localisation_3labels_uterus_test.csv',
                                      run_csv='data_localisation_3labels_uterus_test.csv',
                                      results_dir='/data/project/Localisation/wshop_data/loc3D/results-3D-3lab-loc-cls/',
                                      checkpoint_dir='/data/project/Localisation/wshop_data/loc3D/checkpoints-3D-2lab-loc-cls/',
                                      exp_name='Loc_3D',
                                      task_net='unet_3D',
                                      cls_net='cls_3D',
                                      n_classes=N_classes)

args.gpu_ids = [0]

if args.training:
    print("Training")
    model = md.LocalisationClassificationNetwork3DMultipleLabels(args)

    # Run train
    ####################
    losses_train = model.train(args)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')



if args.testing:
    print("Testing")
    model = md.LocalisationClassificationNetwork3DMultipleLabels(args)

    # Run inference
    ####################
    model.test(args)



if args.running:
    print("Running")
    model = md.LocalisationClassificationNetwork3DMultipleLabels(args)

    # Run inference
    ####################
    model.run(args)
