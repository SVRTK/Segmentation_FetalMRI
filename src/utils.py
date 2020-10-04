
# SVRTK : SVR reconstruction based on MIRTK and CNN-based processing for fetal MRI
#
# Copyright 2018-2020 King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# see the License for the specific language governing permissions and
# limitations under the License.



# ==================================================================================================================== #
#
#     Part of the code adapted from: https://github.com/arnab39/cycleGAN-PyTorch/blob/master/utils.py
#
# ==================================================================================================================== #

import copy
import os
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, distance_transform_cdt, gaussian_filter
import numpy as np
import torch


# To make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


# To save the checkpoint
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


# To print networks
def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')

#
# ==================================================================================================================== #
#

# Plot predictions vs gt - cropped
def plot_img_cropped(patch_size, epoch_,
                     real_img_crop_, fake_img_crop_):
    # Figure
    plt.figure(figsize=(8, 6))

    # # # # # # Real image cropped
    plt.subplot(2, 1, 1)
    # IMG
    img_plot = torch.zeros((patch_size[0], 2 * patch_size[1]))
    for i in range(2):  # body and brain
        img_plot[:, i * patch_size[1]:i * patch_size[1] + patch_size[1]] = \
            real_img_crop_[0, i, :, :].cpu().data
    plt.imshow(img_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    plt.colorbar()
    plt.ylabel('GT Cropped')
    plt.xticks([])
    plt.yticks([])
    plt.title('Cropped images E = ' + str(epoch_ + 1))

    # # # # # # IMG + PRED Segmentation
    plt.subplot(2, 1, 2)
    # IMG
    img_plot = torch.zeros((patch_size[0], 2 * patch_size[1]))
    for i in range(2):  # body and brain
        img_plot[:, i * patch_size[1]:i * patch_size[1] + patch_size[1]] = \
            fake_img_crop_[0, i, :, :].cpu().data
    plt.imshow(img_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    plt.ylabel('PR Cropped')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.show()


# Get cropped brain and body
def get_cropped_brain_body(input_size_, output_size_, img_input_, seg_output_):
    # Get sizes
    b, c, h, w, d = input_size_
    new_h, new_w, new_d = output_size_

    def calculate_centroids(seg_):
        b, c, h, w, d = seg_.shape

        # Calculate centroids for channels 1 -> end (exclude background)
        centroids_ = np.zeros((b, c - 1, 3))
        exists_ = np.zeros((b, c - 1))  # 0 if mask does not exist, 1 if it exists

        # Calculate centre of mass
        coords_x, coords_y, coords_z = np.meshgrid(np.arange(0, h),
                                                   np.arange(0, w),
                                                   np.arange(0, d), indexing='xy')

        def get_coords(lab):
            exists = 1
            if np.sum(lab) == 0:
                coords_x_, coords_y_, coords_z_ = 0, 0, 0
                exists = 0
            else:
                coords_x_ = np.round(np.sum(coords_x * lab) / np.sum(lab))
                coords_y_ = np.round(np.sum(coords_y * lab) / np.sum(lab))
                coords_z_ = np.round(np.sum(coords_z * lab) / np.sum(lab))

            return np.asarray([coords_x_, coords_y_, coords_z_]), exists

        for i in np.arange(0, b):
            for j in np.arange(1, c):
                centroids_[i, j - 1, :], exists_[i, j - 1] = get_coords(seg_[i, j])

        return centroids_, exists_

    # Centroids are n_batches x n_channels [0=placenta,1=body,2=brain]
    centroids, exists_ = calculate_centroids(seg_output_)

    # Create brain and body patches around centre of mass
    img_cropped_input_ = np.zeros((b, 2, new_h, new_w, new_d))
    seg_cropped_output_ = np.zeros((b, 2, new_h, new_w, new_d))
    patch_coords_ = np.zeros((b, 2, 3), dtype='int16')  # b, 0body|1brain
    mask_exists_ = np.zeros((b, 2))                     # b, 0body|1brain

    for i in np.arange(0, b):
        for j in [1, 2]:  # only interested in body and brain
            coords_x = centroids[i, j, 0]  # b, c, 0|1|2
            coords_y = centroids[i, j, 1]  # b, c, 0|1|2
            coords_z = centroids[i, j, 2]  # b, c, 0|1|2

            mask_exists_[i, j-1] = exists_[i, j]

            # Calculate start point of patch
            patch_y = 0 if (int(coords_x - new_h // 2) < 0 or int(coords_x + new_h // 2) >= h) \
                else int(coords_x - new_h // 2)
            patch_x = 0 if (int(coords_y - new_w // 2) < 0 or int(coords_y + new_w // 2) >= w) \
                else int(coords_y - new_w // 2)
            patch_z = 0 if (int(coords_z - new_d // 2) < 0 or int(coords_z + new_d // 2) >= d) \
                else int(coords_z - new_d // 2)

            # Add to list of coordinates
            patch_coords_[i, j - 1, 0] = int(patch_x)
            patch_coords_[i, j - 1, 1] = int(patch_y)
            patch_coords_[i, j - 1, 2] = int(patch_z)

            # Add to variable
            img_cropped_input_[i, j - 1] = img_input_[i, 0, patch_x:patch_x + new_h,
                                           patch_y:patch_y + new_w,
                                           patch_z:patch_z + new_d]
            seg_cropped_output_[i, j - 1] = seg_output_[i, j + 1, patch_x:patch_x + new_h,
                                            patch_y:patch_y + new_w,
                                            patch_z:patch_z + new_d]

    return torch.from_numpy(img_cropped_input_), \
           torch.from_numpy(seg_cropped_output_), \
           patch_coords_, mask_exists_


# Create grid based on patch coordinates
def create_grid(vol_size, batch_size, patch_size, patch_coords, id_c):
    # create empty grid
    vectors = [torch.arange(0, s) for s in vol_size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)            # y, x, z
    grid = torch.unsqueeze(grid, 0)      # add batch
    grid = grid.type(torch.FloatTensor)

    # normalise grid -1, 1
    for i in range(len(vol_size)):
        grid[:, i, ...] = 2 * (grid[:, i, ...] / (vol_size[i] - 1) - 0.5)

    # Repeat for batch size
    grid = torch.cat(batch_size*[grid])

    # # Select part of grid which is around the area for body
    for i_b in range(batch_size):
        if i_b == 0:
            new_grid = grid[i_b:i_b+1, :,
                       patch_coords[i_b, id_c, 0]:patch_coords[i_b, id_c, 0] + patch_size[0],
                       patch_coords[i_b, id_c, 1]:patch_coords[i_b, id_c, 1] + patch_size[1],
                       patch_coords[i_b, id_c, 2]:patch_coords[i_b, id_c, 2] + patch_size[2]]
        else:
            temp_ = grid[i_b:i_b+1, :,
                    patch_coords[i_b, id_c, 0]:patch_coords[i_b, id_c, 0] + patch_size[0],
                    patch_coords[i_b, id_c, 1]:patch_coords[i_b, id_c, 1] + patch_size[1],
                    patch_coords[i_b, id_c, 2]:patch_coords[i_b, id_c, 2] + patch_size[2]]
            new_grid = torch.cat((new_grid, temp_), dim=0)

    # Need to permute as that's how pytorch wants it
    if len(vol_size) == 2:
        new_grid = new_grid.permute(0, 2, 3, 1)
        new_grid = new_grid[..., [1, 0]]
    elif len(vol_size) == 3:
        new_grid = new_grid.permute(0, 2, 3, 4, 1)
        new_grid = new_grid[..., [2, 1, 0]]

    # return grid
    return cuda(new_grid)


def normalise_a_b(data_, a=-1, b=1):
    if torch.is_tensor(data_):
        return (b - a) * (data_ - torch.min(data_)) / (torch.max(data_) - torch.min(data_) + 1e-6) + a
    else:
        return (b - a) * (data_ - np.min(data_)) / (np.max(data_) - np.min(data_) + 1e-6) + a


def apply_distance_transform(lab_):
    '''
    Expected size of image n_batches x n_channels x n_x x n_y
    :param lab_:
    :return:
    '''
    if len(lab_.shape) > 4:
        print('[ERROR] Unimplemented size')
        return None

    nb, nc, _, _ = lab_.shape
    distance_map = np.zeros_like(lab_)

    for i in np.arange(0, nb):
        for j in np.arange(0, nc):
            map_current = distance_transform_edt(lab_[i, j, :, :])
            distance_map[i, j, :, :] = normalise_a_b(map_current, a=-1, b=1)  # * \
            # normalise_a_b(gaussian_filter(lab_[i, j, :, :],
            #                               sigma=(15, 15)),
            #               a=-1, b=1)

    return torch.from_numpy(distance_map)


# Plot predicted and ground truth segmentations together with image
def plot_seg_img(args_, epoch_, seg_gt_, seg_pr_, t2w_gt_):
    # Figure
    plt.figure(figsize=(12, 4))

    # # # # # # IMG + GT Segmentation
    plt.subplot(2, 1, 1)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')

    # SEG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.4)

    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1) + ' ' + args_.exp_name)

    # # # # # # IMG + PRED Segmentation
    plt.subplot(2, 1, 2)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # PRED
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_pr_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_pr_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.4)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.show()


def plot_seg_img_labels(args_, epoch_, seg_gt_, seg_pr_, t2w_gt_):
    # Figure
    plt.figure(figsize=(8, 4))

    # # # # # # IMG + GT Segmentation
    plt.subplot(2, 1, 1)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.n_classes * args_.crop_height))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[0, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # SEG
    seg_plot = torch.zeros((args_.crop_width, args_.n_classes * args_.crop_height))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_gt_[0, i, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.4)
    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1) + ' ' + args_.exp_name)

    # # # # # # IMG + PRED Segmentation
    plt.subplot(2, 1, 2)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.n_classes * args_.crop_height))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[0, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # PRED
    seg_plot = torch.zeros((args_.crop_width, args_.n_classes * args_.crop_height))
    for i in range(seg_pr_.shape[1]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_pr_[0, i, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.4)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.show()


# Plot predicted and ground truth segmentations together with image
def plot_seg_img_map(args_, epoch_, seg_gt_, seg_pr_, map_gt_, map_pr_, t2w_gt_):
    # Figure
    plt.figure(figsize=(12, 8))

    # # # # # # IMG + GT Segmentation
    plt.subplot(4, 1, 1)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # SEG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.6)

    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1) + ' ' + args_.exp_name)

    # # # # # # IMG + PRED Segmentation
    plt.subplot(4, 1, 2)
    # IMG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            t2w_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # PRED
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(seg_pr_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            seg_pr_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.6)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # # # # # # IMG + GT Distance MAP
    plt.subplot(4, 1, 3)
    # SEG
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(map_gt_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            map_gt_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0, cmap='jet') # , alpha=0.6)

    plt.colorbar()
    plt.ylabel('GT Map')
    plt.xticks([])
    plt.yticks([])

    # # # # # # IMG + PRED Distance MAP
    plt.subplot(4, 1, 4)
    # PRED
    seg_plot = torch.zeros((args_.crop_width, args_.batch_size * args_.crop_height))
    for i in range(map_pr_.shape[0]):
        seg_plot[:, i * args_.crop_height:i * args_.crop_height + args_.crop_height] = \
            map_pr_[i, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0, cmap='jet') # , alpha=0.6)
    plt.ylabel('PR Map')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()


    plt.show()


def plot_seg_img_map_one(args_, epoch_, seg_gt_, seg_pr_, map_gt_, map_pr_, t2w_gt_):
    # Figure
    plt.figure(figsize=(12, 8))

    # # # # # # IMG + GT Segmentation
    plt.subplot(2, 2, 1)
    # IMG
    img_plot = t2w_gt_[0, 0, :, :].cpu().data
    plt.imshow(img_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # PRED
    # seg_plot = gaussian_filter(seg_gt_[0, 0, :, :].cpu().data.numpy(), sigma=(5, 5))
    seg_plot = seg_gt_[0, 0, :, :].cpu().data.numpy()
    plt.imshow(seg_plot, vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.6)
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('E = ' + str(epoch_ + 1) + ' ' + args_.exp_name)

    # # # # # # IMG + PRED Segmentation
    plt.subplot(2, 2, 2)
    # IMG
    img_plot = t2w_gt_[0, 0, :, :].cpu().data
    plt.imshow(img_plot.numpy(), vmin=0.0, vmax=1.0, cmap='gray')
    # PRED
    seg_plot = seg_pr_[0, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0, cmap='seismic', alpha=0.6)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # # # # # # IMG + GT Distance MAP
    plt.subplot(2, 2, 3)
    seg_plot = map_gt_[0, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0, cmap='jet')
    plt.colorbar()
    plt.ylabel('GT Map')
    plt.xticks([])
    plt.yticks([])

    # # # # # # IMG + PRED Distance MAP
    plt.subplot(2, 2, 4)
    seg_plot = map_pr_[0, 0, :, :].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0, cmap='jet')
    plt.ylabel('PR Map')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.show()


def plot_losses_train(args, losses_train, title_plot):
    # Get some variables about the train
    ####################
    n_epochs_train = len(losses_train)
    keys_train = list(losses_train[0].keys())
    n_iter_train = len(losses_train[0][keys_train[0]])
    print(keys_train)
    print(len(keys_train))

    # Average losses
    ####################
    losses_train_mean = {key_: [] for key_ in keys_train}
    losses_train_std = {key_: [] for key_ in keys_train}
    for epoch_ in losses_train:
        for key_ in keys_train:
            losses_train_mean[key_].append(np.mean(epoch_[key_]))
            losses_train_std[key_].append(np.std(epoch_[key_]))

    # Plot losses
    ####################
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 18))
    for i_, key_ in enumerate(keys_train):
        plt.subplot(6, 2, i_ + 1)
        plt.fill_between(np.arange(1, n_epochs_train),
                         [x - y for x, y in zip(losses_train_mean[key_][1:],
                                                losses_train_std[key_][1:])],
                         [x + y for x, y in zip(losses_train_mean[key_][1:],
                                                losses_train_std[key_][1:])],
                         alpha=0.2)
        plt.plot(np.arange(0, n_epochs_train), losses_train_mean[key_])
        plt.xlabel('epochs')
        plt.ylabel(key_)
        if i_ == 0:
            # plt.ylim([1e+1, 1e+2])
            plt.title(args.exp_name)

        if i_ >= len(keys_train)-1:
            break

    plt.savefig(args.results_dir + '/' + title_plot + str(n_epochs_train) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.show()



# ==================================================================================================================== #
#
# CLASS FOR TRAIN/TEST ARGUMENTS - can be transformed easily into arguments parser for command line interface
#
# ==================================================================================================================== #
class ArgumentsTrainTestLocalisation():
    def __init__(self,
                 epochs=100,
                 decay_epoch=1,
                 batch_size=2,
                 lr=0.0002,
                 gpu_ids=0,
                 crop_height=128,
                 crop_width=128,
                 crop_depth=128,
                 validation_steps=5,
                 lamda=10.0,
                 lamda2=1.0,
                 training=False,
                 testing=False,
                 running=False,
                 root_dir='/data/projects/localisation/data/',
                 csv_dir='/data/projects/localisation/data/',
                 results_dir='data/project/localisation/network_results/results-3D-2lab-loc/',
                 checkpoint_dir='/data/project/localisation/network_results/checkpoints-3D-2lab-loc/',
                 train_csv='data_localisation_2labels_train.csv',
                 valid_csv='data_localisation_2labels_valid.csv',
                 test_csv='data_localisation_2labels_brain_mixed_test.csv',
                 run_csv='data_localisation_2labels_brain_mixed_run.csv',
                 exp_name='test',
                 task_net='unet_3D',
                 cls_net='cls_3D',
                 n_classes=2):

        self.epochs = epochs
        self.decay_epoch = decay_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.validation_steps = validation_steps
        self.gpu_ids = gpu_ids
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_depth = crop_depth
        self.exp_name = exp_name
        self.lamda = lamda
        self.lamda2 = lamda2
        self.training = training
        self.testing = testing
        self.running = running
        self.csv_dir = csv_dir
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        self.run_csv = run_csv
        self.root_dir = root_dir
        self.task_net = task_net
        self.cls_net = cls_net
        self.n_classes = n_classes

