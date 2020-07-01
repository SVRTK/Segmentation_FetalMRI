#
# Author: Irina Grigorescu
# Date:      02-06-2020
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
#  TRAIN Localisation Network
#
# ==================================================================================================================== #

N_epochs = 2500  # 1940  # 2500


# # # Prepare arguments
args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                      batch_size=8,
                                      lr=0.001,
                                      crop_height=128,
                                      crop_width=128,
                                      crop_depth=64,
                                      validation_steps=8,
                                      lamda=10,
                                      training=True,
                                      testing=True,
                                      root_dir='/data/project/Localisation/localisation-only/',
                                      csv_dir='/data/project/Localisation/',
                                      results_dir='/data/project/Localisation/results/',
                                      checkpoint_dir='/data/project/Localisation/checkpoints/',
                                      norm='instance',
                                      exp_name='Localisation_06_02',
                                      task_net='unet_128',
                                      ntf=64,
                                      n_classes=1,
                                      no_dropout=False)

args.gpu_ids = [0]

if args.training:
    print("Training")
    model = md.LocalisationNetwork(args)

    # Run train
    ####################
    losses_train = model.train(args)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')



if args.testing:
    print("Testing")
    model = md.LocalisationNetwork(args)

    # Run inference
    ####################
    model.test(args)


