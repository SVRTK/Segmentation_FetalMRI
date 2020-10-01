# Localisation and segmentation for fetal MRI -- under development 

This reposity provides a solution for localisation and segmentation in fetal MRI stacks or motion-corrected volumes. 


## Quick how-to (not tested):

	conda env create -f environment.yml
	conda env list
	conda activate Segmentation_FetalMRI


## Quick how-to running train.py

You need to modify:
1. root_dir=<your_own_path>
2. csv_dir=<your_own_path>
3. train_csv='new_data_localisation_train.csv'  # example
4. valid_csv='new_data_localisation_valid.csv'  # example
5. test_csv='new_data_localisation_test.csv'    # example
6. results_dir=<your_own_path>
7. checkpoint_dir=<your_own_path>
...


## Contacts

In case of any questions regarding the code - please report an issue or contact Irina Grigorescu. 


