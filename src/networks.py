
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
#     OPERATIONS and METHODS FROM
#     https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py
#
#     Part of the code adapted from: https://github.com/arnab39/cycleGAN-PyTorch/blob/master/model.py
#
# ==================================================================================================================== #


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
from src.dataloaders import LocalisationDataLoader, ToTensor, RandomCrop3D
from torchvision import transforms

import src.utils as utils
from src.losses import dice_loss, generalised_dice_loss


#
# ==================================================================================================================== #
#

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


def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
                         nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
                         norm_layer(out_dim), nn.ReLU(True))


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super(ResidualBlock, self).__init__()


        refl_pad = nn.ReplicationPad3d(1)
        conv = nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=True)

        res_block = [refl_pad,
                     conv_norm_relu(dim, dim, kernel_size=3,
                                    norm_layer=norm_layer, bias=True)]

        res_block += [nn.Dropout(0.5)]
        
        res_block += [refl_pad,
                      conv,
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)
        
#
# ==================================================================================================================== #
#

class EncoderBlock(nn.Module):
    '''
    Encoder block class
    '''
    def __init__(self, in_channels, out_channels, k_size, pad_size):
        super(EncoderBlock, self).__init__()

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
    def __init__(self, in_channels, middle_channels, out_channels, k_size, pad_size):
        super(DecoderBlock, self).__init__()

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


#
# ==================================================================================================================== #
#

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UNet, self).__init__()
        
        
        self.enc1 = EncoderBlock(in_channels=input_nc, out_channels=32, k_size=3, pad_size=1)
        self.enc2 = EncoderBlock(in_channels=32, out_channels=64, k_size=3, pad_size=1)
        self.enc3 = EncoderBlock(in_channels=64, out_channels=128, k_size=3, pad_size=1)
        self.enc4 = EncoderBlock(in_channels=128, out_channels=256, k_size=3, pad_size=1)
        self.enc5 = EncoderBlock(in_channels=256, out_channels=512, k_size=3, pad_size=1)
        
        self.dec1 = DecoderBlock(in_channels=512, middle_channels=256, out_channels=256, k_size=3, pad_size=1)
        self.dec2 = DecoderBlock(in_channels=256+256, middle_channels=128, out_channels=128, k_size=3, pad_size=1)
        self.dec3 = DecoderBlock(in_channels=128+128, middle_channels=64, out_channels=64, k_size=3, pad_size=1)
        self.dec4 = DecoderBlock(in_channels=64+64, middle_channels=32, out_channels=32, k_size=3, pad_size=1)
        self.dec5 = DecoderBlock(in_channels=32+32, middle_channels=16, out_channels=16, k_size=3, pad_size=1)
        
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.prefinal = ResidualBlock(dim=16, norm_layer=nn.InstanceNorm3d)
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



# ==================================================================================================================== #
#
#  Define localisation architecture
#
# ==================================================================================================================== #

def define_LocalisationNet(input_nc, output_nc, netL, gpu_ids=[0]):
    
    loc_net = UNet(input_nc, output_nc)
    return init_network(loc_net, gpu_ids)



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

        self.n_labels = output_nc - 1

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
            is_augment=False,
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
                                          
                                          
        transformed_dataset_run = LocalisationDataLoader(
            csv_file=args.csv_dir + args.run_csv,
            root_dir=args.root_dir,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([RandomCrop3D(output_size=(args.crop_width,
                                                                    args.crop_height,
                                                                    args.crop_depth),
                                                       is_random=False),
                                          ToTensor()]))

                                          
        self.dataloaders = {
            'train': DataLoader(transformed_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'valid': DataLoader(transformed_dataset_valid, batch_size=1, shuffle=False, num_workers=1),
            'test': DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=1)
            'run': DataLoader(transformed_dataset_run, batch_size=1, shuffle=False, num_workers=1)
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

                    seg_current = data_point['lab']
                    seg_output = []

                    # label 1 - background
                    bg = torch.ones_like(img_input)
                    for l in range(self.n_labels):
                        bg = bg - seg_current[:, [l], ...]
                    seg_output.append(bg)
                    
                    
                    # the rest of the labels
                    for l in range(self.n_labels):
                        seg_output.append(seg_current[:, [l], ...])


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


                        # Plot validation results
                        #######################################################
                        if epoch % 1 == 0 and not plotted:

                            print("....................................................................................")
                            
                            utils.plot_seg_img_labels(args, epoch,
                                                      seg_output[:,:,:,:,args.crop_depth//2],
                                                      seg_pred_val[:,:,:,:,args.crop_depth//2],
                                                      img_input[:,:,:,:,args.crop_depth//2])
                        
                        
                            out_prob = self.Loc(img_input)
                            plt.figure(figsize=(3*(self.n_labels + 1), 3))
                            
                            # plot probabilities
                            for l in range(self.n_labels + 1):
                                plt.subplot(1,(self.n_labels + 1),l)
                                plt.imshow(out_prob.cpu().data.numpy()[0,0,:,:,args.crop_depth//2])
                                plt.colorbar()
                                
                            plt.show()

                            print("....................................................................................")
                        


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

                        # break


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
            

            # Fetch middle slices from the data
            ##################################################
            # Image data
            img_input = Variable(data_point['image'])
            
            
            seg_current = data_point['lab']
            seg_output = []
            
            
            # label 1 - background
            bg = torch.ones_like(img_input)
            for l in range(self.n_labels):
                bg = bg - seg_current[:, [l], ...]
            seg_output.append(bg)
                         
                         
            # the rest of the labels
            for l in range(self.n_labels):
                seg_output.append(seg_current[:, [l], ...])

            

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


            # plot results

            print("....................................................................................")
            
            print(" - ",  i)


            # # # # # # # # # # # # # # # # # # # # # # # #
            img_gt = img_input[0, 0, ...].cpu().data.numpy()
            seg_gt = np.argmax(seg_output[0, :, ...].cpu().data.numpy(), axis=0).astype(int)
            seg_pr = np.argmax(seg_pred_val[0, :, ...].cpu().data.numpy(), axis=0).astype(int)
            out_prob = self.Loc(img_input)
            
            
            
            # # # # # # # # # # # # # # # # # # # # # # # #
            def save_nii_img_seg(args_, name_, img_gt_, seg_gt_, seg_pr_, img_aff_, seg_aff_, ind):

                # Save as nib file - IMG GT
                gt_img = nib.Nifti1Image(img_gt_, img_aff_)
                nib.save(gt_img, args_.results_dir + name_ + '_img-' + str(ind) + '.nii.gz')
                img_tmp_info = nib.load(args_.results_dir + name_ + '_img-' + ii + '.nii.gz')

                # Save as nib file - SEG GT
                gt_lab = nib.Nifti1Image(seg_gt_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(gt_lab, args_.results_dir + name_ + '_seg-' + str(ind) + '.nii.gz')

                # Save as nib file - SEG PR
                pr_lab = nib.Nifti1Image(seg_pr_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(pr_lab, args_.results_dir + name_ + '_seg_pr-' + str(ind) + '.nii.gz')




            # # # # # # # # # # # # # # # # # # # # # # # #
            def save_nii_img_seg_prob(args_, name_, img_gt_, seg_gt_, seg_pr_, img_aff_, seg_aff_, prob_out_, n_labels_, ind):

                # Save as nib file - IMG GT
                gt_img = nib.Nifti1Image(img_gt_, img_aff_)
                nib.save(gt_img, args_.results_dir + name_ + '_img-' + str(ind) + '.nii.gz')
                img_tmp_info = nib.load(args_.results_dir + name_ + '_img-' + ii + '.nii.gz')
                                
                # Save as nib file - SEG GT
                gt_lab = nib.Nifti1Image(seg_gt_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(gt_lab, args_.results_dir + name_ + '_seg-' + str(ind) + '.nii.gz')
                                
                # Save as nib file - SEG PR
                pr_lab = nib.Nifti1Image(seg_pr_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(pr_lab, args_.results_dir + name_ + '_seg_pr-' + str(ind) + '.nii.gz')
                
                # Save probabilities nib file - ...
                for l in range(n_labels_):
                    prob_out = nib.Nifti1Image(prob_out_.cpu().data.numpy()[0,l+1,:,:,:], img_tmp_info.affine, img_tmp_info.header)
                    nib.save(prob_out, args_.results_dir + name_ + '_pr-' +  str(l+1) ' _ ' + str(ind) + '.nii.gz')
                


            name = data_point['name'][0].split('/')[0] + '_' + data_point['name'][0].split('/')[-1]
            img_aff = data_point['img_aff'][0, ...].numpy().astype(np.float32)
            seg_aff = data_point['seg_aff'][0, ...].numpy().astype(np.float32)
            seg_prob = np.argmax(seg_pred_val[0, :, ...].cpu().data.numpy(), axis=0).astype(int)
  
  
            save_nii_img_seg(args, name, img_gt, seg_gt, seg_pr, img_aff, seg_aff, i)
            
  
  
            
            # # # # # # # # # # # # # # # # # # # # # # # #
            def displ_res_all(img_gt_, seg_gt_, seg_pr_, prob_out_, n_labels_, pos_, n_labels_):
            
            
                plt.figure(figsize=((3*(3+n_labels_)), 9))
                
                M=3
                N=3+n_labels_
                
                z=1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, :, pos_],cmap='gray')
                plt.title('XY: ORG')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, :, pos_],cmap='gray')
                plt.imshow(seg_gt_[:, :, pos_], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XY: GT')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, :, pos_],cmap='gray')
                plt.imshow(seg_pr_[:, :, pos_], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XY: PRED')
                plt.colorbar()

                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,:,:,pos_], vmin=0, vmax=100)
                    plt.title('XY: PROB')
                    plt.colorbar()
                
                                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, pos_, :],cmap='gray')
                plt.title('XZ: ORG')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, pos_, :],cmap='gray')
                plt.imshow(seg_gt_[:, pos_, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XZ: GT')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, pos_, :],cmap='gray')
                plt.imshow(seg_pr_[:, pos_, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XZ: PRED')
                plt.colorbar()
                
                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,:,pos_,:], vmin=0, vmax=100)
                    plt.title('XZ: PROB')
                    plt.colorbar()

                

                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[pos_, :, :],cmap='gray')
                plt.title('YZ: ORG')
                plt.colorbar()



                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[pos_, :, :],cmap='gray')
                plt.imshow(seg_gt_[pos_, :, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('YZ: GT')
                plt.colorbar()
                

                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[pos_, :, :],cmap='gray')
                plt.imshow(seg_pr_[pos_, :, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('YZ: PRED')
                plt.colorbar()
                
                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,pos_,:,:], vmin=0, vmax=100)
                    plt.title('YZ: PROB')
                    plt.colorbar()
            
                
                plt.show()
            
            
            
            displ_res_all(img_gt, seg_gt, seg_pr, out_prob, args.crop_depth//2, self.n_labels)
            
        
            print("....................................................................................")





    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def run(self, args):
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
            

            # Fetch middle slices from the data
            ##################################################
            # Image data
            img_input = Variable(data_point['image'])
            
            
#            seg_current = data_point['lab']
                       
            # Create cuda variables:
            img_input = utils.cuda(img_input)


            with torch.no_grad():
                # Forward pass through UNet
                ##################################################
                if args.n_classes == 1:
                    seg_pred_val = torch.sigmoid(self.Loc(img_input))

                    seg_pred_val[seg_pred_val >= 0.5] = 1.0
                    seg_pred_val[seg_pred_val <= 0.5] = 0.0

                else:
                    seg_pred_val = torch.round(torch.softmax(self.Loc(img_input), dim=1))


            # plot results

            print("....................................................................................")
            
            print(" - ",  i)


            # # # # # # # # # # # # # # # # # # # # # # # #
            img_gt = img_input[0, 0, ...].cpu().data.numpy()
            seg_pr = np.argmax(seg_pred_val[0, :, ...].cpu().data.numpy(), axis=0).astype(int)
            out_prob = self.Loc(img_input)
            
            
            
            # # # # # # # # # # # # # # # # # # # # # # # #
            def save_nii_img_seg(args_, name_, img_gt_, seg_pr_, img_aff_, seg_aff_, ind):

                # Save as nib file - IMG GT
                gt_img = nib.Nifti1Image(img_gt_, img_aff_)
                nib.save(gt_img, args_.results_dir + name_ + '_img-' + str(ind) + '.nii.gz')
                img_tmp_info = nib.load(args_.results_dir + name_ + '_img-' + ii + '.nii.gz')

                # Save as nib file - SEG PR
                pr_lab = nib.Nifti1Image(seg_pr_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(pr_lab, args_.results_dir + name_ + '_seg_pr-' + str(ind) + '.nii.gz')




            # # # # # # # # # # # # # # # # # # # # # # # #
            def save_nii_img_seg_prob(args_, name_, img_gt_, seg_pr_, img_aff_, seg_aff_, prob_out_, n_labels_, ind):

                # Save as nib file - IMG GT
                gt_img = nib.Nifti1Image(img_gt_, img_aff_)
                nib.save(gt_img, args_.results_dir + name_ + '_img-' + str(ind) + '.nii.gz')
                img_tmp_info = nib.load(args_.results_dir + name_ + '_img-' + ii + '.nii.gz')
                                
                # Save as nib file - SEG PR
                pr_lab = nib.Nifti1Image(seg_pr_, img_tmp_info.affine, img_tmp_info.header)
                nib.save(pr_lab, args_.results_dir + name_ + '_seg_pr-' + str(ind) + '.nii.gz')
                
                # Save probabilities nib file - ...
                for l in range(n_labels_):
                    prob_out = nib.Nifti1Image(prob_out_.cpu().data.numpy()[0,l+1,:,:,:], img_tmp_info.affine, img_tmp_info.header)
                    nib.save(prob_out, args_.results_dir + name_ + '_pr-' +  str(l+1) ' _ ' + str(ind) + '.nii.gz')
                



            name = data_point['name'][0].split('/')[0] + '_' + data_point['name'][0].split('/')[-1]
            img_aff = data_point['img_aff'][0, ...].numpy().astype(np.float32)
            seg_prob = np.argmax(seg_pred_val[0, :, ...].cpu().data.numpy(), axis=0).astype(int)


  
            save_nii_img_seg(args, name, img_gt, seg_pr, img_aff, seg_aff, i)
            
  
            
            # # # # # # # # # # # # # # # # # # # # # # # #
            def displ_res_all(img_gt_, seg_pr_, prob_out_, n_labels_, pos_, n_labels_):
            
            
                plt.figure(figsize=((3*(2+n_labels_)), 9))
                
                M=3
                N=2+n_labels_
                
                z=1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, :, pos_],cmap='gray')
                plt.title('XY: ORG')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, :, pos_],cmap='gray')
                plt.imshow(seg_pr_[:, :, pos_], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XY: PRED')
                plt.colorbar()

                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,:,:,pos_], vmin=0, vmax=100)
                    plt.title('XY: PROB')
                    plt.colorbar()
                
                     
                     
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, pos_, :],cmap='gray')
                plt.title('XZ: ORG')
                plt.colorbar()
                
                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[:, pos_, :],cmap='gray')
                plt.imshow(seg_pr_[:, pos_, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('XZ: PRED')
                plt.colorbar()
                
                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,:,pos_,:], vmin=0, vmax=100)
                    plt.title('XZ: PROB')
                    plt.colorbar()

                

                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[pos_, :, :],cmap='gray')
                plt.title('YZ: ORG')
                plt.colorbar()

                z=z+1
                plt.subplot(M,N,z)
                plt.imshow(img_gt_[pos_, :, :],cmap='gray')
                plt.imshow(seg_pr_[pos_, :, :], alpha=0.5, vmin=0, vmax=l_num)
                plt.title('YZ: PRED')
                plt.colorbar()
                
                for l in range(n_labels_):
                    z=z+1
                    plt.subplot(M,N,z)
                    plt.imshow(prob_out_.cpu().data.numpy()[0,l+1,pos_,:,:], vmin=0, vmax=100)
                    plt.title('YZ: PROB')
                    plt.colorbar()
            
            
                
                plt.show()
            
            
            
            displ_res_all(img_gt, seg_pr, out_prob, args.crop_depth//2, self.n_labels)
            
        
            print("....................................................................................")






