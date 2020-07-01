# Localisation -- under development -- not for use at the moment

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
t2w,lab
brain-10/2d-res-stack1.nii.gz,brain-10/2d-res-mask1.nii.gz
brain-10/2d-res-stack2.nii.gz,brain-10/2d-res-mask2.nii.gz
brain-10/2d-res-stack3.nii.gz,brain-10/2d-res-mask3.nii.gz
brain-10/2d-res-stack4.nii.gz,brain-10/2d-res-mask4.nii.gz
brain-10/2d-res-stack5.nii.gz,brain-10/2d-res-mask5.nii.gz
...
