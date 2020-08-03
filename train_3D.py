#
# Author: Irina Grigorescu
# Date:      01-07-2020
#
# Train a UNet to localise the baby brain
#
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

N_epochs = 25


# # # Prepare arguments
args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                      batch_size=2,
                                      lr=0.001,
                                      crop_height=128,
                                      crop_width=128,
                                      crop_depth=128,
                                      validation_steps=8,
                                      lamda=10,
                                      training=True,
                                      testing=False,
                                      root_dir='/data/project/Localisation/data/localisation-only/',
                                      csv_dir='/data/project/Localisation/data/',
                                      train_csv='new_data_localisation_train.csv',
                                      valid_csv='new_data_localisation_valid.csv',
                                      test_csv='new_data_localisation_test.csv',
                                      results_dir='/data/project/Localisation/network_results/results-3D-loc/',
                                      checkpoint_dir='/data/project/Localisation/network_results/checkpoints-3D-loc/',
                                      exp_name='Loc_07_01_3D',
                                      task_net='unet_3D',
                                      n_classes=1)

args.gpu_ids = [0]

if args.training:
    print("Training")
    model = md.LocalisationNetwork3D(args)

    # Run train
    ####################
    losses_train = model.train(args)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')



if args.testing:
    print("Testing")
    model = md.LocalisationNetwork3D(args)

    # Run inference
    ####################
    model.test(args)


