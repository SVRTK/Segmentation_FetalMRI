#
# Data preprocessing before training
#
# Author: Irina Grigorescu
# Date:      28-05-2020
#

import numpy as np
import glob
import os
import nibabel as nib
import pandas as pd
from src.constants import FOLDER_PATH_DATA


# Number of stacks:
no_stacks = len(sorted(glob.glob(FOLDER_PATH_DATA + '/*')))

# Total number of files
n_files = 0

TEST_SUBJs = ['brain-1', 'brain-14', 'brain-20']

# CSV Files to populate
csv_file_train = []
csv_file_valid = []
csv_file_test = []


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Go through all the folders
for id_folder, folder_name in enumerate(sorted(glob.glob(FOLDER_PATH_DATA + '/*'))):
    # print folder name
    print('===> ' +
          str(id_folder) + ' ' + folder_name + '/ imgs: ' +
          str(len(sorted(glob.glob(folder_name + '/*mask*')))))

    # Choose an id to keep for validate
    id_valid = id_folder % (len(sorted(glob.glob(folder_name + '/*mask*'))) + 1)
    print('     - Valid_ID:', id_valid)

    # Go through all the files (mask + mri) in each folder
    for id_file, (file_name_mask, file_name_stack) in enumerate(zip(sorted(glob.glob(folder_name + '/*mask*')),
                                                                    sorted(glob.glob(folder_name + '/*stack*')))):
        print('     - ' + file_name_mask, file_name_stack)

        # Read data:
        img_current = nib.load(file_name_stack).get_data().astype(np.float32)
        msk_current = nib.load(file_name_mask).get_data().astype(np.float32)
        img_aff = nib.load(file_name_stack).get_affine()
        lab_aff = nib.load(file_name_mask).get_affine()
        print('     - ', img_current.shape)
        print('     - ', msk_current.shape)

        # Create csv file for training
        current_row = file_name_stack.split('/')[-2] + '/' + file_name_stack.split('/')[-1] + ',' + \
                      file_name_mask.split('/')[-2] + '/' + file_name_mask.split('/')[-1]

        # If brain-i in TEST_SUBJs, append to test csv
        if file_name_stack.split('/')[-2] in TEST_SUBJs:
            csv_file_test.append(current_row)
        # If id_file == id_valid, append to valid csv
        elif id_file == id_valid:
            csv_file_valid.append(current_row)
        # Else append to train csv
        else:
            csv_file_train.append(current_row)

        n_files += 1

print('Total number of files:', n_files)
print('Train data:', len(csv_file_train))
print('Valid data:', len(csv_file_valid))
print('Test  data:', len(csv_file_test))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Save csv files
for data_type, data_csv in zip(['train', 'valid', 'test'],
                               [csv_file_train, csv_file_valid, csv_file_test]):
    id_header = 0
    with open('/data/project/Localisation/new_data_localisation_' + data_type + '.csv', 'w') as f:
        if id_header == 0:
            f.write("t2w,lab\n")
            id_header += 1
        for baby in data_csv:
            f.write("%s\n" % baby)
