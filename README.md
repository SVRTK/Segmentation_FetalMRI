# Localisation for fetal MRI -- under development 

This reposity provides a solution for localisation and segmentation in fetal MRI stacks or motion-corrected volumes. 


## Quick how-to (not tested):

	conda env create -f environment.yml
	conda env list
	conda activate Localisation

## Quick how-to running train.py

You need to modify:
1. root_dir=<your_own_path>
2. csv_dir=<your_own_path>
3. train_csv='new_data_localisation_train.csv'  # example
4. valid_csv='new_data_localisation_valid.csv'  # example
5. test_csv='new_data_localisation_test.csv'    # example
6. results_dir=<your_own_path>
7. checkpoint_dir=<your_own_path>

### Things to know:
* *root_dir* contains your data. This path will be appended in front of the data filenames which are specified in the csv files
* *csv_dir* needs to contain the three specified files:
1. new_data_localisation_train.csv
2. new_data_localisation_valid.csv
3. new_data_localisation_test.csv
* *results_dir* is a path where the results will be stored
* *checkpoint_dir* is a path where the checkpoints will be stored


### Example csv file:
    t2w, lab1, lab2
    resampled-datasets/100067/stack.nii.gz    resampled-datasets/100067/mask-1.nii.gz    resampled-datasets/100067/mask-2.nii.gz
    resampled-datasets/100072/stack.nii.gz    resampled-datasets/100072/mask-1.nii.gz    resampled-datasets/100072/mask-2.nii.gz
    resampled-datasets/100074/stack.nii.gz    resampled-datasets/100074/mask-1.nii.gz    resampled-datasets/100074/mask-2.nii.gz
    resampled-datasets/100078/stack.nii.gz    resampled-datasets/100078/mask-1.nii.gz    resampled-datasets/100078/mask-2.nii.gz
	...


## Contacts

In case of any questions regarding the code - please report an issue or contact Irina Grigorescu. 


