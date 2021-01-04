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

Use prepare-for-cnn function from SVRTK to resample & pad all files to the same grid (128x128x128):

	res=128
	all_num_lab=3
	number_of_stacks=$(ls ${input_file_folder}/*.nii* | wc -l)
	stack_names=$(ls ${input_file_folder}/*.nii*)
	${mirtk_path}/mirtk prepare-for-cnn res-files stack-files train-cnn-files.csv train-info-summary.csv ${res} ${number_of_stacks} $(echo $stack_names)  ${all_num_lab} 0 


## Perform training
 
Modify train_3D_Localisation.py:

    - root_dir=<your_own_path>
    - csv_dir=<your_own_path>
    - train_csv='train-cnn-files.csv'  # see example
    - valid_csv='valid-cnn-files.csv'  
    - test_csv='test-cnn-files.csv'    
    -  run_csv='trun-cnn-files.csv'    
    - results_dir=<your_own_path>
    - checkpoint_dir=<your_own_path>
    - I_size=<res>
    - N_classes=<all_num_lab + 1 (bg_label)>
    - ...


Train CNN:

	python train_3D_Localisation.py




## License

The SVRTK Fetal MRI Segmentation package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0). The license enables usage of SVRTK in both commercial and non-commercial applications, without restrictions on the licensing applied to the combined work.


## Citation and acknowledgements

In case you found SVRTK Fetal MRI Segmentation useful please give appropriate credit to the software by providing the corresponding link to our github repository.
