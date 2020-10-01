
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


def generalised_dice_loss(pred_seg, target_seg, eps=1e-6):
    """
    Generalised Dice Loss
    :param pred_seg:
    :param target_seg:
    :param include_background:
    :param eps:
    :return:
    """
    reduce_axis = list(range(2, len(target_seg.shape)))
    ba_size = target_seg.size(0)
    n_class = target_seg.size(1)

    # 1. Calculate weights
    weights = torch.reciprocal(torch.pow(torch.sum(target_seg, reduce_axis).float(), 2))
    for b in weights:
        infs = torch.isinf(b)
        b[infs] = 0.0
        b[infs] = torch.max(b)
    weights[:, 0] = 0.0
    # print(weights.shape, weights)

    # 2. calculate intersection
    intersection = (pred_seg.view(ba_size, n_class, -1) *
                    target_seg.view(ba_size, n_class, -1)).sum(2)
    intersection = intersection

    # 3. calculate reunion
    reunion = pred_seg.view(ba_size, n_class, -1).sum(2) + \
              target_seg.view(ba_size, n_class, -1).sum(2)
    reunion = reunion

    # print('intersection', intersection.shape, intersection)
    # print('reunion', reunion.shape, reunion)

    return ((2.0 * (intersection * weights).sum(1) + eps) / ((reunion * weights).sum(1) + eps)).mean()

