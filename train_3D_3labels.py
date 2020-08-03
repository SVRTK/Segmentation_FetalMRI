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

N_epochs = 100


# # # Prepare arguments
args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                      batch_size=2,
                                      lr=0.002,
                                      crop_height=128,
                                      crop_width=128,
                                      crop_depth=128,
                                      validation_steps=8,
                                      lamda=10,
                                      training=True,
                                      testing=True,
                                      root_dir='/data/project/Localisation/data/resampled-with-3-labels/',
                                      csv_dir='/data/project/Localisation/data/',
                                      train_csv='data_localisation_3labels_train.csv',
                                      valid_csv='data_localisation_3labels_valid.csv',
                                      test_csv='data_localisation_3labels_test.csv',
                                      results_dir='/data/project/Localisation/network_results/results-3D-3lab-loc/',
                                      checkpoint_dir='/data/project/Localisation/network_results/checkpoints-3D-3lab-loc/',
                                      exp_name='Loc_08_03_3D',
                                      task_net='unet_3D',
                                      n_classes=4)

args.gpu_ids = [0]

if args.training:
    print("Training")
    model = md.LocalisationNetwork3DMultipleLabels(args)

    # Run train
    ####################
    losses_train = model.train(args)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')



if args.testing:
    print("Testing")
    model = md.LocalisationNetwork3DMultipleLabels(args)

    # Run inference
    ####################
    model.test(args)


