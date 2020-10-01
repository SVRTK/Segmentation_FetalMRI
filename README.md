# Localisation and segmentation for fetal MRI -- under development 

Solutions for localisation and segmentation in fetal MRI stacks or motion-corrected volumes. 


## Quick how-to: setup

	conda env create -f environment.yml
	conda env list
	conda activate Segmentation_FetalMRI


## Quick how-to: running train_3D_Segmentation.py

You need to modify:
1. root_dir=<your_own_path>
2. csv_dir=<your_own_path>
3. train_csv='data_localisation_2labels_train.csv'  # example
4. valid_csv='data_localisation_2labels_valid.csv'  # example
5. test_csv='data_localisation_2labels_test.csv'    # example
6. run_csv='data_localisation_2labels_run.csv'    # example
7. results_dir=<your_own_path>
8. checkpoint_dir=<your_own_path>
...


## Contacts

In case of any questions regarding the code - please report an issue or contact Irina Grigorescu. 



