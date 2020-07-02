#
# Author: Irina Grigorescu
# Date:      02-06-2020
#
# Utility functions for training
#
# Some code from:
#      https://github.com/arnab39/cycleGAN-PyTorch/blob/master/utils.py
#

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


# ==================================================================================================================== #
#
#  My code:
#
# ==================================================================================================================== #
def normalise_a_b(data_, a=-1, b=1):
    return (b-a) * (data_ - np.min(data_)) / (np.max(data_) - np.min(data_) + 1e-6) + a


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
                 batch_size=1,
                 lr=0.0002,
                 gpu_ids=0,
                 crop_height=256,
                 crop_width=256,
                 crop_depth=256,
                 validation_steps=5,
                 lamda=10.0,
                 lamda2=1.0,
                 training=False,
                 testing=False,
                 root_dir='/data/project/dHCP_data_str4cls/3_resampled_rig/',
                 csv_dir='/home/igr18/Work/PycharmProjects/DomainAdaptationSeg/data/',
                 results_dir='/data/project/PIPPI2020/DACycleGAN/results/',
                 checkpoint_dir='/data/project/PIPPI2020/DACycleGAN/checkpoints/',
                 train_csv='new_data_localisation_train.csv',
                 valid_csv='new_data_localisation_valid.csv',
                 test_csv='new_data_localisation_test.csv',
                 norm='instance',
                 exp_name='test',
                 task_net='unet_2D',
                 n_classes=1):

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
        self.csv_dir = csv_dir
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        self.norm = norm
        self.root_dir = root_dir
        self.task_net = task_net
        self.n_classes = n_classes
