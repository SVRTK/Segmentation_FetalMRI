#
# Author: Irina Grigorescu
# Date:      01-07-2020
#
# Test some code
#

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, distance_transform_cdt, gaussian_filter


example_image = nib.load('/data/project/Localisation/localisation-only/brain-1/2d-res-mask1.nii.gz')
example_image = example_image.get_fdata()
example_image = example_image[:, :, 40]


plt.figure(figsize=(12,12))

plt.subplot(2,2,1)
plt.imshow(example_image)
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(distance_transform_edt(example_image))
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(distance_transform_cdt(example_image))
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(gaussian_filter(distance_transform_cdt(example_image), sigma=(5, 5)))
plt.colorbar()

plt.show()