# Localisation and segmentation for fetal MRI 

3D-Unet based solutions for localisation and segmentation in fetal MRI stacks or motion-corrected volumes. 

![GitHub Logo](whole-body-btfe-example.png)

## Contacts

In case of any questions regarding the code - please report an issue or contact Irina Grigorescu. 


## Setup

	conda env create -f environment.yml
	conda env list
	conda activate Segmentation_FetalMRI


## Prepare datasets

Use prepare-for-cnn function from SVRTK to resample & pad all files to the same grid (128x128x128) for training, validation and testing:

	res=128
	all_num_lab=3
	
	train_number_of_stacks=$(ls ${input_train_file_folder}/*.nii* | wc -l)
	train_stack_names=$(ls ${input_train_file_folder}/*.nii*)
	train_mask_names=$(ls ${input_train__mask_file_folder}/*.nii*)
	${mirtk_path}/mirtk prepare-for-cnn train-res-files train-stack-files train-cnn-files.csv train-info-summary.csv ${res} ${train_number_of_stacks} $(echo $train_stack_names)  ${all_num_lab} 1 $(echo $train_mask_names)
	
	valid_number_of_stacks=$(ls ${input_valid_file_folder}/*.nii* | wc -l)
	valid_stack_names=$(ls ${input_valid_file_folder}/*.nii*)
	valid_mask_names=$(ls ${input_valid_mask_file_folder}/*.nii*)
	${mirtk_path}/mirtk prepare-for-cnn valid-res-files valid-stack-files valid-cnn-files.csv valid-info-summary.csv ${res} ${valid_number_of_stacks} $(echo $valid_stack_names)  ${all_num_lab} 1 $(echo $valid_mask_names)

	test_number_of_stacks=$(ls ${input_test_file_folder}/*.nii* | wc -l)
	test_stack_names=$(ls ${input_test_file_folder}/*.nii*)
	test_mask_names=$(ls ${input_test_mask_file_folder}/*.nii*)
	${mirtk_path}/mirtk prepare-for-cnn test-res-files test-stack-files test-cnn-files.csv test-info-summary.csv ${res} ${test_number_of_stacks} $(echo $test_stack_names)  ${all_num_lab} 1 $(echo $test_mask_names)
	
	run_number_of_stacks=$(ls ${input_run_file_folder}/*.nii* | wc -l)
	run_stack_names=$(ls ${input_run_file_folder}/*.nii*)
	${mirtk_path}/mirtk prepare-for-cnn run-res-files run-stack-files run-cnn-files.csv run-info-summary.csv ${res} ${run_number_of_stacks} $(echo $run_stack_names)  ${all_num_lab} 0 




## Perform training
 
Modify train_3D_Localisation.py:

    - root_dir=<your_own_path>
    - csv_dir=<your_own_path>
    - train_csv='train-cnn-files.csv'  # see example
    - valid_csv='valid-cnn-files.csv'  
    - test_csv='test-cnn-files.csv'    
    -  run_csv='run-cnn-files.csv'    
    - results_dir=<your_own_path>
    - checkpoint_dir=<your_own_path>
    - I_size=<res>
    - N_classes=<all_num_lab + 1 (bg_label)>
    - ...


Train CNN and run testing:

	python train_3D_Localisation.py




## License

The SVRTK Fetal MRI Segmentation package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0). The license enables usage of SVRTK in both commercial and non-commercial applications, without restrictions on the licensing applied to the combined work.


## Citation and acknowledgements

In case you found SVRTK Fetal MRI Segmentation useful please give appropriate credit to the software by providing the corresponding link to our github repository and the following reference to the describing implementation of the network: 

> Uus, Alena, Irina Grigorescu, Milou van Poppel, Emer Hughes, Johannes Steinweg, Thomas Roberts, David Lloyd, Kuberan Pushparajah, and Maria Deprez. 2021. “3D UNet with GAN Discriminator for Robust Localisation of the Fetal Brain and Trunk in MRI with Partial Coverage of the Fetal Body.” BioRxiv.  https://doi.org/10.1101/2021.06.23.449574.

