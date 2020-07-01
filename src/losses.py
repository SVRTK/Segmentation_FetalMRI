#
# Losses
#
# Author: Irina Grigorescu
# Date:      28-05-2020
#


import torch


def dice_loss(pred_seg, target_seg, eps=1e-6):
    """
    Dice Loss
    :param pred_seg:
    :param target_seg:
    :param include_background:
    :param eps:
    :return:
    """
    ba_size = target_seg.size(0)
    n_class = target_seg.size(1)

    # 2. calculate intersection
    intersection = (pred_seg.view(ba_size, n_class, -1) *
                    target_seg.view(ba_size, n_class, -1)).sum(2)
    intersection = intersection

    # 3. calculate reunion
    reunion = pred_seg.view(ba_size, n_class, -1).sum(2) + \
              target_seg.view(ba_size, n_class, -1).sum(2)
    reunion = reunion

    return ((2.0 * intersection.sum(1) + eps) / (reunion.sum(1) + eps)).mean()
