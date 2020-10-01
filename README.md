# Localisation and segmentation for fetal MRI -- under development 

Solutions for localisation and segmentation in fetal MRI stacks or motion-corrected volumes. 


## Quick how-to: setup

	conda env create -f environment.yml
	conda env list
	conda activate Segmentation_FetalMRI


## Quick how-to: running train_3D_Segmentation.py

1. Prepare the datasets for 3D localisation - resampled stacks and the corresponding binary label masks

    i_size=128
    
    mkdir resampled_datasets/${case_id}
    
    mirtk pad-3d original_datasets/${case_id}/stack.nii.gz resampled_datasets/${case_id}/stack.nii.gz ${i_size} 1
    
    mirtk pad-3d original_datasets/${case_id}/mask-1.nii.gz resampled_datasets/${case_id}/mask-1.nii.gz ${i_size} 0
    
    mirtk pad-3d original_datasets/${case_id}/mask-2.nii.gz resampled_datasets/${case_id}/mask-2.nii.gz ${i_size} 0
    
    ...

2. Prepare .csv files for training, validation, testing and running

    t2w                                                               lab1                                                                 lab2
    
    resampled-datasets/100027/stack.nii.gz    resampled-datasets/100027/mask-1.nii.gz    resampled-datasets/100027/mask-2.nii.gz
    
    resampled-datasets/100034/stack.nii.gz    resampled-datasets/100034/mask-1.nii.gz    resampled-datasets/100034/mask-2.nii.gz
    
    resampled-datasets/100037/stack.nii.gz    resampled-datasets/100037/mask-1.nii.gz    resampled-datasets/100037/mask-2.nii.gz
    
    ...

3. Modify train_3D_Segmentation.py:

    - root_dir=<your_own_path>
    - csv_dir=<your_own_path>
    - train_csv='data_localisation_2labels_train.csv'  # example
    - valid_csv='data_localisation_2labels_valid.csv'  # example
    - test_csv='data_localisation_2labels_test.csv'    # example
    -  run_csv='data_localisation_2labels_run.csv'    # example
    - results_dir=<your_own_path>
    - checkpoint_dir=<your_own_path>
    - I_size=<image_size>
    - N_classes=<number_of_labels+1(bg_label)>
    - ...



## Contacts

In case of any questions regarding the code - please report an issue or contact Irina Grigorescu. 



