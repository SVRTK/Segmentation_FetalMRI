#
# Networks
#
# Author: Irina Grigorescu
# Date:      28-05-2020
#
# Networks
#   Some code adapted from:         https://github.com/arnab39/cycleGAN-PyTorch/blob/master/model.py
#

import itertools
import functools
import os
import time
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.dataloaders import LocalisationDataLoader, ToTensor, RandomCrop2D
from torchvision import transforms

import src.utils as utils
from src.losses import dice_loss

# ==================================================================================================================== #
#
#  OPERATIONS and METHODS FROM
#     https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py
#
# ==================================================================================================================== #
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

    print('Network initialized with kaiming_normal_.')
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2,True))


def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


# ==================================================================================================================== #
#
#  U-NET ARCHITECTURE FROM https://github.com/arnab39/cycleGAN-PyTorch/blob/master/arch/generators.py
#
# ==================================================================================================================== #
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_tanh=True):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if use_tanh:
                up = [nn.ReLU(True), upconv, nn.Tanh()]
            else:
                up = [nn.ReLU(True), upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, use_tanh=True):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer, use_tanh=use_tanh)
        self.unet_model = unet_block

    def forward(self, input):
        # input = nn.ReplicationPad2d(8)(input)
        return self.unet_model(input)


# ==================================================================================================================== #
#
#  Define localisation architecture
#
# ==================================================================================================================== #
def define_LocalisationNet(input_nc, output_nc, ntf, netL, norm='batch', use_dropout=True, use_tanh=True, gpu_ids=[0]):
    norm_layer = get_norm_layer(norm_type=norm)

    if netL == 'unet_128':
        loc_net = UnetGenerator(input_nc, output_nc, 5, ntf,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_tanh=use_tanh)
    elif netL == 'unet_256':
        loc_net = UnetGenerator(input_nc, output_nc, 6, ntf,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_tanh=use_tanh)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % netL)

    return init_network(loc_net, gpu_ids)


# ==================================================================================================================== #
#
#  Define training class for localisation
#
# ==================================================================================================================== #
class LocalisationNetwork(object):
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Loc = define_LocalisationNet(input_nc=1,
                                          output_nc=args.n_classes,
                                          ntf=args.ntf,
                                          netL=args.task_net,
                                          norm=args.norm,
                                          use_dropout=not args.no_dropout,
                                          use_tanh=False,
                                          gpu_ids=args.gpu_ids)

        utils.print_networks([self.Loc], ['Loc'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.DL = dice_loss

        # Optimizers
        #####################################################
        self.l_optimizer = torch.optim.Adam(self.Loc.parameters(), lr=args.lr)

        self.l_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.l_optimizer,
                                                                base_lr=args.lr / 1000,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)


        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Loc.load_state_dict(ckpt['Loc'])
            self.l_optimizer.load_state_dict(ckpt['l_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.losses_train = []

        # Loaders
        #####################################################
        transformed_dataset_train = LocalisationDataLoader(
            csv_file=args.csv_dir + '/new_data_localisation_train.csv',
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_valid = LocalisationDataLoader(
            csv_file=args.csv_dir + '/new_data_localisation_valid.csv',
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_test = LocalisationDataLoader(
            csv_file=args.csv_dir + '/new_data_localisation_test.csv',
            root_dir=args.root_dir,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=False),
                                          ToTensor()]))

        self.dataloaders = {
            'train': DataLoader(transformed_dataset_train, batch_size=1,
                                shuffle=True, num_workers=4),
            'valid': DataLoader(transformed_dataset_valid, batch_size=1,
                                shuffle=True, num_workers=4),
            'test': DataLoader(transformed_dataset_test, batch_size=1,
                               shuffle=False, num_workers=1)
        }



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_localisation_loss = 1e10

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.l_optimizer.param_groups[0]['lr']
            print('LEARNING RATE = %.7f' % lr)

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'loc_loss_train': [], 'loc_loss_valid': [], 'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train', 'valid']:

                for i, data_point in enumerate(self.dataloaders[phase]):

                    # step
                    len_dataloader = len(self.dataloaders[phase])
                    step = epoch * len_dataloader + i + 1

                    # Fetch some slices from the data
                    ##################################################
                    # Image data
                    _, _, _, _, nz = data_point['image'].shape
                    img_input = torch.zeros((args.batch_size, 1, args.crop_width, args.crop_height))
                    seg_output = torch.zeros((args.batch_size, 1, args.crop_width, args.crop_height))

                    # Concatenate for batches because z is different all the time
                    for j in np.arange(0, args.batch_size):
                        z_select = np.random.choice(np.arange(0, nz))

                        # Select slices
                        img_input_slice = data_point['image'][:, :, :, :, z_select]
                        lab_input_slice = data_point['lab'][:, :, :, :, z_select]

                        # Create batch
                        img_input[j, :, :, :] = img_input_slice
                        seg_output[j, :, :, :] = lab_input_slice

                    img_input = utils.cuda(Variable(img_input))
                    seg_output = utils.cuda(Variable(seg_output))

                    # TRAIN
                    ##################################################
                    if phase == 'train':
                        ##################################################
                        # Set optimiser to zero grad
                        ##################################################
                        self.l_optimizer.zero_grad()

                        # Forward pass through network
                        ##################################################
                        seg_pred = self.Loc(img_input)
                        # print(seg_pred.shape)

                        # Dice Loss
                        ###################################################
                        loc_loss = (1. - self.DL(torch.sigmoid(seg_pred), seg_output)) * args.lamda

                        # Store metrics
                        metrics['loc_loss_train'].append(loc_loss.item())

                        # Update generators & segmentation
                        ###################################################
                        loc_loss.backward()
                        self.l_optimizer.step()


                    # VALIDATE
                    #######################################################
                    else:
                        self.Loc.eval()

                        with torch.no_grad():
                            # Forward pass through UNet
                            ##################################################
                            seg_pred_val = self.Loc(img_input)

                            # Dice Loss
                            ###################################################
                            loc_loss = (1. - self.DL(torch.sigmoid(seg_pred_val), seg_output)) * args.lamda

                            # Store metrics
                            metrics['loc_loss_valid'].append(loc_loss.item())

                        # Save best
                        #######################################################
                        if best_localisation_loss >= loc_loss and epoch > 0:

                            # Localisation
                            best_localisation_loss = loc_loss.item()
                            print("Best Localisation Valid Loss %.2f" % (best_localisation_loss))

                            # Override the latest checkpoint for best generator loss
                            utils.save_checkpoint({'epoch': epoch + 1,
                                                   'Loc': self.Loc.state_dict(),
                                                   'l_optimizer': self.l_optimizer.state_dict()},
                                                  '%s/latest_best_loss_E%d.ckpt' % (args.checkpoint_dir, epoch + 1))

                            # Plot some images
                            if epoch % 5 == 0 and not plotted:
                                plotted = True
                                utils.plot_seg_img(args, epoch, seg_output, seg_pred_val, img_input)

                        # Stop early -- Don't go through all the validation set
                        if i >= args.validation_steps:
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("%s Epoch: (%3d) (%5d/%5d) (%3d) | Loc Loss:%.2e | %.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           loc_loss, time_elapsed // 60, time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Loc': self.Loc.state_dict(),
                                   'l_optimizer': self.l_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.l_lr_scheduler.step()

        return self.losses_train


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def test(self, args):
        """
        Inference
        :param args:
        :return:
        """
        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        self.start_epoch = ckpt['epoch']
        self.Loc.load_state_dict(ckpt['Loc'])

        self.Loc.eval()

        # Inference - go through each test data
        #####################################################
        for i, data_point in enumerate(self.dataloaders['test']):

            # Fetch some slices from the data
            ##################################################
            # Image data
            _, _, nx, ny, nz = data_point['image'].shape
            img_gt = np.zeros((nx, ny, nz))
            seg_gt = np.zeros((nx, ny, nz))
            seg_pr = np.zeros((nx, ny, nz))

            # Go through all the slices
            for z_select in np.arange(0, nz):
                img_input = utils.cuda(Variable(data_point['image'][:, :, :, :, z_select]))
                seg_output = utils.cuda(Variable(data_point['lab'][:, :, :, :, z_select]))

                with torch.no_grad():
                    # Forward pass through UNet
                    ##################################################
                    seg_pred_val = torch.sigmoid(self.Loc(img_input))
                    seg_pred_val[seg_pred_val >= 0.5] = 1.0
                    seg_pred_val[seg_pred_val <= 0.5] = 0.0

                # Store here:
                img_gt[:, :, z_select] = img_input[0, 0, :, :].cpu().data.numpy()
                seg_gt[:, :, z_select] = seg_output[0, 0, :, :].cpu().data.numpy()
                seg_pr[:, :, z_select] = seg_pred_val[0, 0, :, :].cpu().data.numpy()

            # Save the entire image
            def save_nii_img_seg(args_, name_, img_gt_, seg_gt_, seg_pr_, img_aff_, seg_aff_):
                # Save as nib file - IMG GT
                gt_img = nib.Nifti1Image(img_gt_, img_aff_)
                nib.save(gt_img, args_.results_dir + name_ + '_img' + '.nii.gz')

                # Save as nib file - SEG GT
                gt_lab = nib.Nifti1Image(seg_gt_, seg_aff_)
                nib.save(gt_lab, args_.results_dir + name_ + '_seg' + '.nii.gz')

                # Save as nib file - SEG PR
                pr_lab = nib.Nifti1Image(seg_pr_, seg_aff_)
                nib.save(pr_lab, args_.results_dir + name_ + '_seg_pr' + '.nii.gz')

            name = data_point['name'][0].split('/')[0] + '_' + data_point['name'][0].split('/')[1]
            img_aff = data_point['img_aff'][0, :, :].numpy().astype(np.float32)
            seg_aff = data_point['seg_aff'][0, :, :].numpy().astype(np.float32)
            # print(img_aff.shape, seg_aff.shape)
            save_nii_img_seg(args, name, img_gt, seg_gt, seg_pr, img_aff, seg_aff)

            # break
