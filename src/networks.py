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
from functools import partial

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.dataloaders import LocalisationDataLoader, ToTensor, RandomCrop2D, RandomCrop3D
from torchvision import transforms

import src.utils as utils
from src.losses import dice_loss, generalised_dice_loss

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
                                 norm_layer = nn.BatchNorm2d, bias = False, is_2D=True):
    if is_2D:
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
            norm_layer(out_dim), nn.ReLU(True))
    else:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
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


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, is_2D=True):
        super(ResidualBlock, self).__init__()

        if is_2D:
            refl_pad = nn.ReflectionPad2d(1)
            conv = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
            
        else:
            refl_pad = nn.ReplicationPad3d(1)
            conv = nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias)


        res_block = [refl_pad,
                     conv_norm_relu(dim, dim, kernel_size=3,
                                    norm_layer=norm_layer, bias=use_bias, is_2D=is_2D)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [refl_pad,
                      conv,
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


# ==================================================================================================================== #
#
#  2D  U-NET ARCHITECTURE
#
# ==================================================================================================================== #
class EncoderBlock(nn.Module):
    '''
    Encoder block class
    '''
    def __init__(self, in_channels, out_channels, k_size, pad_size, is_2D=True):
        super(EncoderBlock, self).__init__()

        if is_2D:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.IN1 = nn.InstanceNorm2d(out_channels)
            self.IN2 = nn.InstanceNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.IN1 = nn.InstanceNorm3d(out_channels)
            self.IN2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.IN1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.IN2(self.conv2(x)), inplace=True)
        return x


class DecoderBlock(nn.Module):
    '''
    Decoder block class
    '''
    def __init__(self, in_channels, middle_channels, out_channels, k_size, pad_size, is_2D=True):
        super(DecoderBlock, self).__init__()

        if is_2D:
            self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=k_size, padding=pad_size)
            self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.IN1 = nn.InstanceNorm2d(out_channels)
            self.IN2 = nn.InstanceNorm2d(out_channels)
            self.upsample = partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=k_size, padding=pad_size)
            self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=k_size, padding=pad_size)
            self.IN1 = nn.InstanceNorm3d(out_channels)
            self.IN2 = nn.InstanceNorm3d(out_channels)
            self.upsample = partial(F.interpolate, scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = F.leaky_relu(self.IN1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.IN2(self.conv2(x)), inplace=True)
        x = self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, is_2D=True):
        super(UNet, self).__init__()

        self.enc1 = EncoderBlock(in_channels=input_nc, out_channels=32, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc2 = EncoderBlock(in_channels=32, out_channels=64, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc3 = EncoderBlock(in_channels=64, out_channels=128, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc4 = EncoderBlock(in_channels=128, out_channels=256, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc5 = EncoderBlock(in_channels=256, out_channels=512, k_size=3, pad_size=1, is_2D=is_2D)

        self.dec1 = DecoderBlock(in_channels=512, middle_channels=256, out_channels=256,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec2 = DecoderBlock(in_channels=256+256, middle_channels=128, out_channels=128,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec3 = DecoderBlock(in_channels=128+128, middle_channels=64, out_channels=64,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec4 = DecoderBlock(in_channels=64+64, middle_channels=32, out_channels=32,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec5 = DecoderBlock(in_channels=32+32, middle_channels=16, out_channels=16,
                                 k_size=3, pad_size=1, is_2D=is_2D)

        if is_2D:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.prefinal = ResidualBlock(dim=16, norm_layer=nn.InstanceNorm2d,
                                          use_dropout=True, use_bias=True, is_2D=True)
            self.final = nn.Conv2d(16, output_nc, kernel_size=1)
        else:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.prefinal = ResidualBlock(dim=16, norm_layer=nn.InstanceNorm3d,
                                          use_dropout=True, use_bias=True, is_2D=False)
            self.final = nn.Conv3d(16, output_nc, kernel_size=1)

        self.dropout = nn.Dropout(0.5)


    def forward(self, input):
        enc1 = self.avgpool(self.enc1(input))                # B x  32 x 64 x 64
        enc2 = self.avgpool(self.enc2(enc1))                 # B x  64 x 32 x 32
        enc3 = self.dropout(self.maxpool(self.enc3(enc2)))   # B x 128 x 16 x 16
        enc4 = self.dropout(self.maxpool(self.enc4(enc3)))   # B x 256 x  8 x  8
        enc5 = self.maxpool(self.enc5(enc4))                 # B x 512 x  4 x  4

        dec1 = self.dec1(enc5)                              # B x 256 x   8 x   8
        dec2 = self.dec2(torch.cat([dec1, enc4], dim=1))    # B x 128 x  16 x  16
        dec3 = self.dec3(torch.cat([dec2, enc3], dim=1))    # B x  64 x  32 x  32
        dec4 = self.dec4(torch.cat([dec3, enc2], dim=1))    # B x  32 x  64 x  64
        dec5 = self.dec5(torch.cat([dec4, enc1], dim=1))    # B x  16 x 128 x 128

        final = self.final(self.prefinal(dec5))

        return final


class UNet_DT(nn.Module):
    def __init__(self, input_nc, output_nc, is_2D=True):
        super(UNet_DT, self).__init__()

        self.enc1 = EncoderBlock(in_channels=input_nc, out_channels=32, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc2 = EncoderBlock(in_channels=32, out_channels=64, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc3 = EncoderBlock(in_channels=64, out_channels=128, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc4 = EncoderBlock(in_channels=128, out_channels=256, k_size=3, pad_size=1, is_2D=is_2D)
        self.enc5 = EncoderBlock(in_channels=256, out_channels=512, k_size=3, pad_size=1, is_2D=is_2D)

        self.dec1 = DecoderBlock(in_channels=512, middle_channels=256, out_channels=256,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec2 = DecoderBlock(in_channels=256+256, middle_channels=128, out_channels=128,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec3 = DecoderBlock(in_channels=128+128, middle_channels=64, out_channels=64,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec4 = DecoderBlock(in_channels=64+64, middle_channels=32, out_channels=32,
                                 k_size=3, pad_size=1, is_2D=is_2D)
        self.dec5 = DecoderBlock(in_channels=32+32, middle_channels=16, out_channels=16,
                                 k_size=3, pad_size=1, is_2D=is_2D)

        self.tanh = nn.Tanh()

        if is_2D:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features_block = nn.Conv2d(16, output_nc, kernel_size=1)
            self.final = nn.Conv2d(2, output_nc, kernel_size=1)
        else:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features_block = nn.Conv3d(16, output_nc, kernel_size=1)
            self.final = nn.Conv3d(2, output_nc, kernel_size=1)

    def forward(self, input):
        enc1 = self.avgpool(self.enc1(input))  # B x  32 x 64 x 64
        enc2 = self.avgpool(self.enc2(enc1))   # B x  64 x 32 x 32
        enc3 = self.maxpool(self.enc3(enc2))   # B x 128 x 16 x 16
        enc4 = self.maxpool(self.enc4(enc3))   # B x 256 x  8 x  8
        enc5 = self.maxpool(self.enc5(enc4))   # B x 512 x  4 x  4

        dec1 = self.dec1(enc5)                            # B x 256 x   8 x   8
        dec2 = self.dec2(torch.cat([dec1, enc4], dim=1))  # B x 128 x  16 x  16
        dec3 = self.dec3(torch.cat([dec2, enc3], dim=1))  # B x  64 x  32 x  32
        dec4 = self.dec4(torch.cat([dec3, enc2], dim=1))  # B x  32 x  64 x  64
        dec5 = self.dec5(torch.cat([dec4, enc1], dim=1))  # B x  16 x 128 x 128

        # Features
        features = self.features_block(dec5)

        # Distance map
        distance_map = self.tanh(features)

        # Concatenate distance map and features
        features_distance_map_concat = torch.cat((features, distance_map), dim=1)

        # Final convolution to get segmentation map
        final = self.final(features_distance_map_concat)

        return distance_map, final


# ==================================================================================================================== #
#
#  Define localisation architecture
#
# ==================================================================================================================== #
def define_LocalisationNet(input_nc, output_nc, netL, gpu_ids=[0]):
    if netL == 'unet_2D_DT':
        loc_net = UNet_DT(input_nc, output_nc, is_2D=True)

    elif netL == 'unet_2D':
        loc_net = UNet(input_nc, output_nc, is_2D=True)

    elif netL == 'unet_3D_DT':
        loc_net = UNet_DT(input_nc, output_nc, is_2D=False)

    elif netL == 'unet_3D':
        loc_net = UNet(input_nc, output_nc, is_2D=False)

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
                                          netL=args.task_net,
                                          gpu_ids=args.gpu_ids)

        utils.print_networks([self.Loc], ['Loc'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.DL = dice_loss

        # Optimizers
        #####################################################
        self.l_optimizer = torch.optim.Adam(self.Loc.parameters(), lr=args.lr)

        self.l_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.l_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Create folders if not existing
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
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
            csv_file=args.csv_dir + args.train_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_valid = LocalisationDataLoader(
            csv_file=args.csv_dir + args.valid_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=False,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_test = LocalisationDataLoader(
            csv_file=args.csv_dir + args.test_csv,
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
                        img_input_slice = utils.normalise_a_b(data_point['image'][:, :, :, :, z_select], a=0, b=1)
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
                            seg_pred_val = torch.sigmoid(self.Loc(img_input))

                            # Dice Loss
                            ###################################################
                            loc_loss = (1. - self.DL(seg_pred_val, seg_output)) * args.lamda

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
                        #######################################################
                        if epoch % 10 == 0 and not plotted:
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
        # Try loading checkpoint
        #####################################################
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Loc.load_state_dict(ckpt['Loc'])
        except:
            print('[ERROR] Could not find checkpoint!')

        # Set network to evaluation mode
        #####################################################
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


# ==================================================================================================================== #
#
#  Define training class for localisation - 3D
#
# ==================================================================================================================== #
class LocalisationNetwork3D(object):
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Loc = define_LocalisationNet(input_nc=1,
                                          output_nc=args.n_classes,
                                          netL=args.task_net,
                                          gpu_ids=args.gpu_ids)

        utils.print_networks([self.Loc], ['Loc'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.DL = dice_loss

        # Optimizers
        #####################################################
        self.l_optimizer = torch.optim.Adam(self.Loc.parameters(), lr=args.lr)

        self.l_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.l_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Create folders if not existing
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
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
            csv_file=args.csv_dir + args.train_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_valid = LocalisationDataLoader(
            csv_file=args.csv_dir + args.valid_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=False,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_test = LocalisationDataLoader(
            csv_file=args.csv_dir + args.test_csv,
            root_dir=args.root_dir,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=False),
                                          ToTensor()]))

        self.dataloaders = {
            'train': DataLoader(transformed_dataset_train, batch_size=args.batch_size,
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

            # For each epoch set the validation losses to 0
            loc_loss_valid = 0.0

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
                    img_input = utils.cuda(Variable(data_point['image']))
                    seg_output = utils.cuda(Variable(data_point['lab']))

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
                        if args.n_classes == 1:
                            loc_loss = (1. - self.DL(torch.sigmoid(seg_pred), seg_output)) * args.lamda
                        else:
                            loc_loss = (1. - self.DL(torch.softmax(seg_pred, dim=1), seg_output)) * args.lamda

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
                            if args.n_classes == 1:
                                seg_pred_val = torch.sigmoid(self.Loc(img_input))
                            else:
                                seg_pred_val = torch.softmax(self.Loc(img_input), dim=1)

                            # Dice Loss
                            ###################################################
                            loc_loss = (1. - self.DL(seg_pred_val, seg_output)) * args.lamda

                            # Store metrics
                            metrics['loc_loss_valid'].append(loc_loss.item())

                            # Save valid losses here:
                            loc_loss_valid += loc_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % 10 == 0 and not plotted:
                            plotted = True
                            utils.plot_seg_img(args, epoch,
                                               seg_output[:,:,:,:,args.crop_depth//2],
                                               seg_pred_val[:,:,:,:,args.crop_depth//2],
                                               img_input[:,:,:,:,args.crop_depth//2])


                        # Save best after all validation steps
                        #######################################################
                        if i >= (args.validation_steps - 1):
                            loc_loss_valid /= args.validation_steps

                            print('AVG LOC LOSS VALID | ', loc_loss_valid)

                            # Save best
                            if best_localisation_loss > loc_loss_valid and epoch > 0:

                                # Localisation
                                best_localisation_loss = loc_loss_valid
                                print("Best Localisation Valid Loss %.2f" % (best_localisation_loss))

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Loc': self.Loc.state_dict(),
                                                       'l_optimizer': self.l_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d' % (epoch + 1))

                            # Stop early -- Don't go through all the validation set
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
        # Try loading checkpoint
        #####################################################
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Loc.load_state_dict(ckpt['Loc'])
        except:
            print('[ERROR] Could not find checkpoint!')

        # Set network to evaluation mode
        #####################################################
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


# ==================================================================================================================== #
#
#  Define training class for localisation - 3D
#
# ==================================================================================================================== #
class LocalisationNetwork3DMultipleLabels(object):
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Loc = define_LocalisationNet(input_nc=1,
                                          output_nc=args.n_classes,
                                          netL=args.task_net,
                                          gpu_ids=args.gpu_ids)

        utils.print_networks([self.Loc], ['Loc'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.DL = dice_loss
        self.GDL = generalised_dice_loss

        # Optimizers
        #####################################################
        self.l_optimizer = torch.optim.Adam(self.Loc.parameters(), lr=args.lr)

        self.l_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.l_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Create folders if not existing
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
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
            csv_file=args.csv_dir + args.train_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_valid = LocalisationDataLoader(
            csv_file=args.csv_dir + args.valid_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=False,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_test = LocalisationDataLoader(
            csv_file=args.csv_dir + args.test_csv,
            root_dir=args.root_dir,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=False),
                                          ToTensor()]))

        self.dataloaders = {
            'train': DataLoader(transformed_dataset_train, batch_size=args.batch_size,
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

            # For each epoch set the validation losses to 0
            loc_loss_valid = 0.0

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
                    img_input = Variable(data_point['image'])

                    # I need to fiddle with the labels because they are not one-hot encoded
                    seg_current = data_point['lab']
                    seg_output = []

                    # label 1 - background
                    seg_output.append(torch.ones_like(img_input) - seg_current[:, [0], ...])
                    # label 2 - uterus
                    seg_output.append(seg_current[:, [0], ...] - seg_current[:, [1], ...] - seg_current[:, [2], ...])
                    # label 3 - body
                    seg_output.append(seg_current[:, [1], ...])
                    # label 4 - brain
                    seg_output.append(seg_current[:, [2], ...])

                    # Create cuda variables:
                    img_input = utils.cuda(img_input)
                    seg_output = utils.cuda(torch.cat(seg_output, dim=1))

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
                        if args.n_classes == 1:
                            loc_loss = (1. - self.DL(torch.sigmoid(seg_pred), seg_output)) * args.lamda
                        else:
                            loc_loss = (1. - self.GDL(torch.softmax(seg_pred, dim=1), seg_output)) * args.lamda

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
                            if args.n_classes == 1:
                                seg_pred_val = torch.sigmoid(self.Loc(img_input))
                            else:
                                seg_pred_val = torch.softmax(self.Loc(img_input), dim=1)

                            # Dice Loss
                            ###################################################
                            if args.n_classes == 1:
                                loc_loss = (1. - self.DL(seg_pred_val, seg_output)) * args.lamda
                            else:
                                loc_loss = (1. - self.GDL(seg_pred_val, seg_output)) * args.lamda

                            # Store metrics
                            metrics['loc_loss_valid'].append(loc_loss.item())

                            # Save valid losses here:
                            loc_loss_valid += loc_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % 1 == 0 and not plotted:
                            plotted = True

                            utils.plot_seg_img_labels(args, epoch,
                                                      seg_output[:,:,:,:,args.crop_depth//2],
                                                      seg_pred_val[:,:,:,:,args.crop_depth//2],
                                                      img_input[:,:,:,:,args.crop_depth//2])


                        # Save best after all validation steps
                        #######################################################
                        if i >= (args.validation_steps - 1):
                            loc_loss_valid /= args.validation_steps

                            print('AVG LOC LOSS VALID | ', loc_loss_valid)

                            # Save best
                            if best_localisation_loss > loc_loss_valid and epoch > 0:

                                # Localisation
                                best_localisation_loss = loc_loss_valid
                                print("Best Localisation Valid Loss %.2f" % (best_localisation_loss))

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Loc': self.Loc.state_dict(),
                                                       'l_optimizer': self.l_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Loss: %d' % (epoch + 1, best_localisation_loss))

                            # Stop early -- Don't go through all the validation set
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
        # Try loading checkpoint
        #####################################################
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Loc.load_state_dict(ckpt['Loc'])
        except:
            print('[ERROR] Could not find checkpoint!')

        # Set network to evaluation mode
        #####################################################
        self.Loc.eval()

        distances_centroids = []

        # Inference - go through each test data
        #####################################################
        for i, data_point in enumerate(self.dataloaders['test']):

            # Fetch some slices from the data
            ##################################################
            # Image data
            img_input = Variable(data_point['image'])

            # I need to fiddle with the labels because they are not one-hot encoded
            seg_current = data_point['lab']
            seg_output = []

            # label 1 - background
            seg_output.append(torch.ones_like(img_input) - seg_current[:, [0], ...])
            # label 2 - uterus
            seg_output.append(seg_current[:, [0], ...] - seg_current[:, [1], ...] - seg_current[:, [2], ...])
            # label 3 - body
            seg_output.append(seg_current[:, [1], ...])
            # label 4 - brain
            seg_output.append(seg_current[:, [2], ...])

            # Create cuda variables:
            img_input = utils.cuda(img_input)
            seg_output = utils.cuda(torch.cat(seg_output, dim=1))

            with torch.no_grad():
                # Forward pass through UNet
                ##################################################
                if args.n_classes == 1:
                    seg_pred_val = torch.sigmoid(self.Loc(img_input))

                    seg_pred_val[seg_pred_val >= 0.5] = 1.0
                    seg_pred_val[seg_pred_val <= 0.5] = 0.0

                else:
                    seg_pred_val = torch.round(torch.softmax(self.Loc(img_input), dim=1))

            # Store here:
            # # # # # # # # # # # # # # # # # # # # # # # #
            img_gt = img_input[0, 0, ...].cpu().data.numpy()
            seg_gt = np.argmax(seg_output[0, :, ...].cpu().data.numpy(), axis=0).astype(int)
            seg_pr = np.argmax(seg_pred_val[0, :, ...].cpu().data.numpy(), axis=0).astype(int)

            # Save the entire image
            # # # # # # # # # # # # # # # # # # # # # # # #
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

            name = data_point['name'][0].split('/')[0] + '_' + data_point['name'][0].split('/')[-1]
            img_aff = data_point['img_aff'][0, ...].numpy().astype(np.float32)
            seg_aff = data_point['seg_aff'][0, ...].numpy().astype(np.float32)
            # print(img_aff.shape, seg_aff.shape)
            save_nii_img_seg(args, name, img_gt, seg_gt, seg_pr, img_aff, seg_aff)

            # Calculate centroids
            # # # # # # # # # # # # # # # # # # # # # # # #
            def calculate_centroids(args_, img_gt_, seg_gt_, seg_pr_):
                h, w, d = img_gt_.shape

                centroids_gt = []
                centroids_pr = []

                for i in np.arange(1, args_.n_classes):
                    # Calculate centre of mass
                    coords_x, coords_y, coords_z = np.meshgrid(np.arange(0, w),
                                                               np.arange(0, h),
                                                               np.arange(0, d))

                    def get_coords(lab):
                        coords_x_ = np.round(np.sum(coords_x * lab[[i], ...]) / np.sum(lab[[i], ...]))
                        coords_y_ = np.round(np.sum(coords_y * lab[[i], ...]) / np.sum(lab[[i], ...]))
                        coords_z_ = np.round(np.sum(coords_z * lab[[i], ...]) / np.sum(lab[[i], ...]))

                        return [coords_x_, coords_y_, coords_z_]

                    centroids_gt.append(get_coords(seg_gt_))
                    centroids_pr.append(get_coords(seg_pr_))

                return centroids_gt, centroids_pr

            centroids_gt, centroids_pr = calculate_centroids(args, img_gt,
                                                             seg_output[0, ...].cpu().data.numpy(),
                                                             seg_pred_val[0, ...].cpu().data.numpy())

            def calculate_euclidean_distance_centroids(centroids_gt_, centroids_pr_):
                n_labels = len(centroids_gt_)

                distances = []

                for i in np.arange(0, n_labels):
                    dist = np.sqrt(np.power(centroids_gt_[i][0] - centroids_pr_[i][0], 2) +
                                   np.power(centroids_gt_[i][1] - centroids_pr_[i][1], 2) +
                                   np.power(centroids_gt_[i][2] - centroids_pr_[i][2], 2))
                    distances.append(dist)

                return distances

            distances = calculate_euclidean_distance_centroids(centroids_gt, centroids_pr)
            distances_centroids.append({'name': name,
                                        'lab_dist': distances})

            # Plot centroids
            # # # # # # # # # # # # # # # # # # # # # # # #
            def plot_centroids(img_gt_, seg_gt_, seg_pr_, centroids_gt_, centroids_pr_):
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img_gt_[:, :, 64])
                plt.imshow(seg_gt_[:, :, 64], alpha=0.5)

                plt.plot(centroids_gt_[0][0], centroids_gt_[0][1], '*r')
                plt.plot(centroids_gt_[1][0], centroids_gt_[1][1], '*r')
                plt.plot(centroids_gt_[2][0], centroids_gt_[2][1], '*r')
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.imshow(img_gt_[:, :, 64])
                plt.imshow(seg_pr_[:, :, 64], alpha=0.5)

                plt.plot(centroids_pr_[0][0], centroids_pr_[0][1], '*b')
                plt.plot(centroids_pr_[1][0], centroids_pr_[1][1], '*b')
                plt.plot(centroids_pr_[2][0], centroids_pr_[2][1], '*b')
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.imshow(img_gt_[:, :, 64])
                plt.imshow(seg_pr_[:, :, 64], alpha=0.5)

                plt.plot(centroids_pr_[0][0], centroids_pr_[0][1], '*b')
                plt.plot(centroids_pr_[1][0], centroids_pr_[1][1], '*b')
                plt.plot(centroids_pr_[2][0], centroids_pr_[2][1], '*b')

                plt.plot(centroids_gt_[0][0], centroids_gt_[0][1], '*r')
                plt.plot(centroids_gt_[1][0], centroids_gt_[1][1], '*r')
                plt.plot(centroids_gt_[2][0], centroids_gt_[2][1], '*r')
                plt.colorbar()

                plt.show()

            if i % 10 == 0:
                plot_centroids(img_gt, seg_gt, seg_pr, centroids_gt, centroids_pr)

            # break

        # Save centroids
        # # # # # # # # # # # # # # # # # # # # # # # #
        with open(args.results_dir + 'Centroids.csv', 'w') as f:
            f.write('name,{}\n'.format(','.join(['uterus', 'body', 'brain'])))
            for item in distances_centroids:
                subj_name = item['name']
                distances_current = item['lab_dist']
                f.write('{},{}\n'.format(subj_name, ','.join([str(xx_) for xx_ in distances_current])))



# ==================================================================================================================== #
#
#  Define training class for localisation
#
# ==================================================================================================================== #
class LocalisationNetworkDT(object):
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Loc = define_LocalisationNet(input_nc=1,
                                          output_nc=args.n_classes,
                                          netL=args.task_net,
                                          gpu_ids=args.gpu_ids)

        utils.print_networks([self.Loc], ['Loc'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.SmoothL1 = nn.SmoothL1Loss()
        self.DL = dice_loss

        # Optimizers
        #####################################################
        self.l_optimizer = torch.optim.Adam(self.Loc.parameters(), lr=args.lr)

        self.l_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.l_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Create folders if not existing
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
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
            csv_file=args.csv_dir + args.train_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_valid = LocalisationDataLoader(
            csv_file=args.csv_dir + args.valid_csv,
            root_dir=args.root_dir,
            shuffle=True,
            is_augment=False,
            transform=transforms.Compose([RandomCrop2D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=True),
                                          ToTensor()]))
        transformed_dataset_test = LocalisationDataLoader(
            csv_file=args.csv_dir + args.test_csv,
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
        best_distancemap_loss = 1e10

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.l_optimizer.param_groups[0]['lr']
            print('LEARNING RATE = %.7f' % lr)

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'loc_loss_train': [], 'loc_loss_valid': [],
                       'l1_loss_train': [], 'l1_loss_valid': [], 'lr': [lr]}

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
                    map_output = torch.zeros((args.batch_size, 1, args.crop_width, args.crop_height))

                    # Concatenate for batches because z is different all the time
                    for j in np.arange(0, args.batch_size):
                        z_select = np.random.choice(np.arange(0, nz))

                        # Select slices
                        img_input_slice = utils.normalise_a_b(data_point['image'][:, :, :, :, z_select], a=0, b=1)
                        lab_input_slice = data_point['lab'][:, :, :, :, z_select]
                        map_input_slice = utils.apply_distance_transform(lab_input_slice)

                        # Create batch
                        img_input[j, :, :, :] = img_input_slice
                        seg_output[j, :, :, :] = lab_input_slice
                        map_output[j, :, :, :] = map_input_slice

                    img_input = utils.cuda(Variable(img_input))
                    seg_output = utils.cuda(Variable(seg_output))
                    map_output = utils.cuda(Variable(map_output))

                    # TRAIN
                    ##################################################
                    if phase == 'train':
                        ##################################################
                        # Set optimiser to zero grad
                        ##################################################
                        self.l_optimizer.zero_grad()

                        # Forward pass through network
                        ##################################################
                        map_pred, seg_pred = self.Loc(img_input)
                        # print(seg_pred.shape)

                        # Dice Loss
                        ###################################################
                        loc_loss = (1. - self.DL(torch.sigmoid(seg_pred), seg_output)) * args.lamda

                        # L1 Loss
                        ###################################################
                        l1_loss = self.SmoothL1(map_pred, map_output) * args.lamda2

                        # Total Loss:
                        ###################################################
                        total_loss = loc_loss + l1_loss

                        # Store metrics
                        metrics['loc_loss_train'].append(loc_loss.item())
                        metrics['l1_loss_train'].append(l1_loss.item())

                        # Update generators & segmentation
                        ###################################################
                        total_loss.backward()
                        self.l_optimizer.step()


                    # VALIDATE
                    #######################################################
                    else:
                        self.Loc.eval()

                        with torch.no_grad():
                            # Forward pass through UNet
                            ##################################################
                            map_pred_val, seg_pred_val = self.Loc(img_input)

                            # Dice Loss
                            ###################################################
                            loc_loss = (1. - self.DL(torch.sigmoid(seg_pred_val), seg_output)) * args.lamda

                            # L1 Loss
                            ################################################### # .repeat(1, args.ntf, 1, 1)
                            l1_loss = self.SmoothL1(map_pred_val, map_output) * args.lamda2

                            # Store metrics
                            metrics['loc_loss_valid'].append(loc_loss.item())
                            metrics['l1_loss_valid'].append(l1_loss.item())

                        # Save best
                        #######################################################
                        if best_localisation_loss >= loc_loss and best_distancemap_loss >= l1_loss and epoch > 0:

                            # Localisation
                            best_localisation_loss = loc_loss.item()
                            print("Best Localisation Valid Loss %.2f" % (best_localisation_loss))

                            # Distance Map
                            best_distancemap_loss = loc_loss.item()
                            print("Best Distance Map Valid Loss %.2f" % (best_distancemap_loss))

                            # Override the latest checkpoint for best generator loss
                            utils.save_checkpoint({'epoch': epoch + 1,
                                                   'Loc': self.Loc.state_dict(),
                                                   'l_optimizer': self.l_optimizer.state_dict()},
                                                  '%s/latest_best_loss_E%d.ckpt' % (args.checkpoint_dir, epoch + 1))

                        # Plot some images
                        #######################################################
                        if epoch % 10 == 0 and not plotted:
                            plotted = True
                            utils.plot_seg_img_map(args, epoch, seg_output, seg_pred_val,
                                                   map_output, map_pred_val, img_input)
                            utils.plot_seg_img_map_one(args, epoch, seg_output[[0]], seg_pred_val[[0]],
                                                       map_output[[0]], map_pred_val[[0]], img_input[[0]])

                        # Stop early -- Don't go through all the validation set
                        if i >= args.validation_steps:
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("%s Epoch: (%3d) (%5d/%5d) (%3d) | Loc Loss:%.2e | Map Loss:%.2e |%.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           loc_loss, l1_loss, time_elapsed // 60, time_elapsed % 60))

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

        # Try loading checkpoint
        #####################################################
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Loc.load_state_dict(ckpt['Loc'])
        except:
            print('[ERROR] Could not find checkpoint!')

        # Set network to evaluation mode
        #####################################################
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
                    seg_pred_val = torch.sigmoid(self.Loc(img_input)[1])
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
