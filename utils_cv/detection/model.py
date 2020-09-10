# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import itertools
import json
from typing import Callable, List, Tuple, Union, Generator, Optional, Dict

from pathlib import Path
import shutil

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import torchvision

from .references.engine import train_one_epoch, evaluate
from .references.coco_eval import CocoEvaluator
from .references.pycocotools_cocoeval import compute_ap
from .bbox import bboxes_iou, DetectionBbox
from ..common.gpu import torch_device
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN






from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor



import torch
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align

#from . import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple
import math

from torchvision.models.detection import _utils as det_utils

def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils import one_hot_embedding
from torch.autograd import Variable





class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        #print ('gamma is {}'.format(self.gamma))
        self.alpha = alpha
      #  if isinstance(alpha,(torch.float,torch.int,torch.long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()






def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, loss='f'):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    
    #print ("fastrcnn is called")

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    
    N, num_classes = class_logits.shape
    
  #  print ('num class is {}'.format(num_classes))

    classification_loss_ce = F.cross_entropy(class_logits, labels)
    
    #weights = torch.tensor([1/10000, 1/10000, 1/1000, 1/1000, 1/500]).float().to(labels.device)
    #criterion = nn.CrossEntropyLoss(weight=weights)
    
    #classification_loss_ce_weighted = criterion(class_logits, labels)
    
    
  #  print ("focal loss started to be called")
    if loss == 'f':
        classification_loss = FocalLoss(gamma=5).forward(class_logits, labels)
        
    else:
        classification_loss = F.cross_entropy(class_logits, labels)
    
  #  print ("focal loss number is {}".format(classification_loss))
  #  print ("cross entropy loss number is {}".format(classification_loss_ce))
  #  print ("weighted cross entropy loss number is {}".format(classification_loss_ce_weighted))
        
        
    if  classification_loss == float('inf') or classification_loss is None:
        print (class_logits)
        print (labels)
    

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


def _onnx_heatmaps_to_keypoints(maps, maps_i, roi_map_width, roi_map_height,
                                widths_i, heights_i, offset_x_i, offset_y_i):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)

    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height

    roi_map = F.interpolate(
        maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode='bicubic', align_corners=False)[:, 0]

    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

    x_int = (pos % w)
    y_int = ((pos - x_int) // w)

    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * \
        width_correction.to(dtype=torch.float32)
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * \
        height_correction.to(dtype=torch.float32)

    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones((xy_preds_i_1.shape), dtype=torch.float32)
    xy_preds_i = torch.stack([xy_preds_i_0.to(dtype=torch.float32),
                              xy_preds_i_1.to(dtype=torch.float32),
                              xy_preds_i_2.to(dtype=torch.float32)], 0)

    # TODO: simplify when indexing without rank will be supported by ONNX
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = roi_map.index_select(1, y_int.to(dtype=torch.int64)) \
        .index_select(2, x_int.to(dtype=torch.int64)).view(-1).index_select(0, ind.to(dtype=torch.int64))

    return xy_preds_i, end_scores_i


@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(maps, rois, widths_ceil, heights_ceil,
                                     widths, heights, offset_x, offset_y, num_keypoints):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)

    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(maps, maps[i],
                                                               widths_ceil[i], heights_ceil[i],
                                                               widths[i], heights[i],
                                                               offset_x[i], offset_y[i])
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32),
                              xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat((end_scores.to(dtype=torch.float32),
                                end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0)
    return xy_preds, end_scores


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(maps, rois,
                                                                widths_ceil, heights_ceil, widths, heights,
                                                                offset_x, offset_y,
                                                                torch.scalar_tensor(num_keypoints, dtype=torch.int64))
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = (pos - x_int) // w
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            kp, proposals_per_image, discretization_size
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
    ]
    return im_mask


def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = (box[2] - box[0] + one)
    h = (box[3] - box[1] + one)
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]):(y_1 - box[1]),
                           (x_0 - box[0]):(x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0,
                          unpaded_im_mask.to(dtype=torch.float32),
                          zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0,
                         concat_0,
                         zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes,
                                               torch.scalar_tensor(im_h, dtype=torch.int64),
                                               torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]
    res = [
        paste_mask_in_image(m[0], b, im_h, im_w)
        for m, b in zip(masks, boxes)
    ]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class RoIHeadsNew(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 ):
        super(RoIHeadsNew, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        if self.has_mask():
            assert all(["masks" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses





def customized__init__(self, backbone, num_classes=None,
             # transform parameters
             min_size=800, max_size=1333,
             image_mean=None, image_std=None,
             # RPN parameters
             rpn_anchor_generator=None, rpn_head=None,
             rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
             rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
             rpn_nms_thresh=0.7,
             rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
             rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
             # Box parameters
             box_roi_pool=None, box_head=None, box_predictor=None,
             box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
             box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
             box_batch_size_per_image=512, box_positive_fraction=0.25,
             bbox_reg_weights=None,
             roi_heads=RoIHeadsNew):

    if not hasattr(backbone, "out_channels"):
        raise ValueError(
            "backbone should contain an attribute out_channels "
            "specifying the number of output channels (assumed to be the "
            "same for all the levels)")

    assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
    assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

    if num_classes is not None:
        if box_predictor is not None:
            raise ValueError("num_classes should be None when box_predictor is specified")
    else:
        if box_predictor is None:
            raise ValueError("num_classes should not be None when box_predictor "
                             "is not specified")

    out_channels = backbone.out_channels

    if rpn_anchor_generator is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
    if rpn_head is None:
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

    if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)
        
    if roi_heads is None:

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        
    else:
        print ('need a new roi_heads')
        roi_heads = roi_heads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        
        



    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
    


#torchvision.models.detection.roi_heads.fastrcnn_loss = _fastrcnn_loss

FasterRCNN.__init__ = customized__init__


def _extract_od_results(
    pred: Dict[str, np.ndarray],
    labels: List[str],
    im_path: Union[str, Path] = None,
) -> Dict:
    """ Gets the bounding boxes, masks and keypoints from the prediction object.

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
            or MaskRCNN model, detached in the form of numpy array
        labels: list of labels without "__background__".
        im_path: the image path of the preds

    Return:
        a dict of DetectionBboxes, masks and keypoints
    """
    pred_labels = pred["labels"].tolist()
    pred_boxes = pred["boxes"].tolist()
    pred_scores = pred["scores"].tolist()

    det_bboxes = []
    for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
        label_name = labels[label - 1]
        det_bbox = DetectionBbox.from_array(
            box,
            score=score,
            label_idx=label,
            label_name=label_name,
            im_path=im_path,
        )
        det_bboxes.append(det_bbox)

    out = {"det_bboxes": det_bboxes, "im_path": im_path}

    if "masks" in pred:
        out["masks"] = pred["masks"].squeeze(1)

    if "keypoints" in pred:
        out["keypoints"] = pred["keypoints"]

    return out


def _apply_threshold(
    pred: Dict[str, np.ndarray], threshold: Optional[float] = 0.5
) -> Dict:
    """ Return prediction results that are above the threshold if any.

    Args:
        pred: the output of passing in an image to torchvision's FasterRCNN
            or MaskRCNN model, detached in the form of numpy array
        threshold: iou threshold for a positive detection. Note: set
            threshold to None to omit a threshold
    """
    # apply score threshold
    if threshold:
        selected = pred["scores"] > threshold
        pred = {k: v[selected] for k, v in pred.items()}
    # apply mask threshold
    if "masks" in pred:
        pred["masks"] = pred["masks"] > 0.5
    return pred


def _get_pretrained_rcnn(
    model_func: Callable[..., nn.Module],
    # transform parameters
    min_size: int = 800,
    max_size: int = 1333,
    # RPN parameters
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    # Box parameters
    box_score_thresh: int = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        model_func: pretrained R-CNN model generating functions, such as
            fasterrcnn_resnet50_fpn(), get_pretrained_fasterrcnn(), etc.
        min_size: minimum size of the image to be rescaled before feeding it to the backbone
        max_size: maximum size of the image to be rescaled before feeding it to the backbone
        rpn_pre_nms_top_n_train: number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test: number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train: number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test: number of proposals to keep after applying NMS during testing
        rpn_nms_thresh: NMS threshold used for postprocessing the RPN proposals

    Returns
        The pre-trained model
    """
    model = model_func(
        pretrained=True,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )
    return model


def _tune_box_predictor(model: nn.Module, num_classes: int) -> nn.Module:
    """ Tune box predictor in the model. """
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    # that has num_classes which is based on the dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _tune_mask_predictor(model: nn.Module, num_classes: int) -> nn.Module:
    """ Tune mask predictor in the model. """
    # get the number of input features of mask predictor from the pretrained model
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features, 256, num_classes
    )
    return model


def get_pretrained_fasterrcnn(num_classes: int = None, **kwargs) -> nn.Module:
    """ Gets a pretrained FasterRCNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If None, 91 as COCO datasets.

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    """
    # TODO - reconsider that num_classes includes background. This doesn't feel
    #     intuitive.

    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(fasterrcnn_resnet50_fpn, **kwargs)
    print (model)

    # if num_classes is specified, then create new final bounding box
    # prediction layers, otherwise use pre-trained layers
    if num_classes:
        model = _tune_box_predictor(model, num_classes)

    return model


def get_pretrained_maskrcnn(num_classes: int = None, **kwargs) -> nn.Module:
    """ Gets a pretrained Mask R-CNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If None, 91 as COCO datasets.

    Returns
        The model to fine-tine/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(maskrcnn_resnet50_fpn, **kwargs)

    # if num_classes is specified, then create new final bounding box
    # and mask prediction layers, otherwise use pre-trained layers
    if num_classes:
        model = _tune_box_predictor(model, num_classes)
        model = _tune_mask_predictor(model, num_classes)

    return model


def get_pretrained_keypointrcnn(
    num_classes: int = None, num_keypoints: int = None, **kwargs
) -> nn.Module:
    """ Gets a pretrained Keypoint R-CNN model

    Args:
        num_classes: number of output classes of the model (including the
            background).  If none of num_classes and num_keypoints below are
            not specified, the pretrained model will be returned.
        num_keypoints: number of keypoints
    Returns
        The model to fine-tune/inference with

    For a list of all parameters see:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/keypoint_rcnn.py

    """
    # load a model pre-trained on COCO
    model = _get_pretrained_rcnn(keypointrcnn_resnet50_fpn, **kwargs)

    if num_classes:
        model = _tune_box_predictor(model, num_classes)

    # tune keypoints predictor in the model
    if num_keypoints:
        # get the number of input features of keypoint predictor from the pretrained model
        in_features = (
            model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        )
        # replace the keypoint predictor with a new one
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_features, num_keypoints
        )

    return model


def _calculate_ap(
    e: CocoEvaluator, 
    iou_thres: float = None,
    area_range: str ='all',
    max_detections: int = 100,
    mode: int = 1,
) -> Dict[str, float]:
    """ Calculate the average precision/recall for differnt IoU ranges.

    Args:
        iou_thres: IoU threshold (options: value in [0.5, 0.55, 0.6, ..., 0.95] or None to average over that range)
        area_range: area size range of the target (options: ['all', 'small', 'medium', 'large'])
        max_detections: maximum number of detection frames in a single image (options: [1, 10, 100])
        mode: set to 1 for average precision and otherwise returns average recall
    """
    ap = {}
    for key in e.coco_eval:
        ap[key] = compute_ap(e.coco_eval[key], iouThr=iou_thres, areaRng=area_range, maxDets=max_detections, ap=mode)

    return ap


def _im_eval_detections(
    iou_threshold: float,
    score_threshold: float,
    gt_bboxes: List[DetectionBbox],
    det_bboxes: List[DetectionBbox],
):
    """ Count number of wrong detections and number of missed objects for a single image """
    # Remove all detections with confidence score below a certain threshold
    if score_threshold is not None:
        det_bboxes = [
            bbox for bbox in det_bboxes if bbox.score > score_threshold
        ]

    # Image level statistics.
    # Store (i) if image has at least one missing ground truth; (ii) if image has at least one incorrect detection.
    im_missed_gt = False
    im_wrong_det = False

    # Object level statistics.
    # Store (i) if ground truth objects were found; (ii) if detections are correct.
    found_gts = [False] * len(gt_bboxes)
    correct_dets = [False] * len(det_bboxes)

    # Check if any object was detected in an image
    if len(det_bboxes) == 0:
        if len(gt_bboxes) > 0:
            im_missed_gt = True

    else:
        # loop over ground truth objects and all detections for a given image
        for gt_index, gt_bbox in enumerate(gt_bboxes):
            gt_label = gt_bbox.label_name

            for det_index, det_bbox in enumerate(det_bboxes):
                det_label = det_bbox.label_name
                iou_overlap = bboxes_iou(gt_bbox, det_bbox)

                # mark as good if detection has same label as the ground truth,
                # and if the intersection-over-union area is above a threshold
                if gt_label == det_label and iou_overlap >= iou_threshold:
                    found_gts[gt_index] = True
                    correct_dets[det_index] = True

        # Check if image has at least one wrong detection, or at least one missing ground truth
        im_wrong_det = min(correct_dets) == 0
        if len(gt_bboxes) > 0 and min(found_gts) == 0:
            im_missed_gt = True

    # Count
    obj_missed_gt = len(found_gts) - np.sum(found_gts)
    obj_wrong_det = len(correct_dets) - np.sum(correct_dets)
    return im_wrong_det, im_missed_gt, obj_wrong_det, obj_missed_gt


def ims_eval_detections(
    detections: List[Dict],
    data_ds: Subset,
    detections_neg: List[Dict] = None,
    iou_threshold: float = 0.5,
    score_thresholds: List[float] = np.linspace(0, 1, 51),
):
    """ Count number of wrong detections and number of missed objects for multiple image """
    # get detection bounding boxes and corresponding ground truth for all images
    det_bboxes_list = [d["det_bboxes"] for d in detections]
    gt_bboxes_list = [
        data_ds.dataset.anno_bboxes[d["idx"]] for d in detections
    ]

    # Get counts for test images
    out = [
        [
            _im_eval_detections(
                iou_threshold,
                score_threshold,
                gt_bboxes_list[i],
                det_bboxes_list[i],
            )
            for i in range(len(det_bboxes_list))
        ]
        for score_threshold in score_thresholds
    ]
    out = np.array(out)
    im_wrong_det_counts = np.sum(out[:, :, 0], 1)
    im_missed_gt_counts = np.sum(out[:, :, 1], 1)
    obj_wrong_det_counts = np.sum(out[:, :, 2], 1)
    obj_missed_gt_counts = np.sum(out[:, :, 3], 1)

    # Count how many images have either a wrong detection or a missed ground truth
    im_error_counts = np.sum(np.max(out[:, :, 0:2], 2), 1)

    # Get counts for negative images
    if detections_neg:
        neg_scores = [
            [box.score for box in d["det_bboxes"]] for d in detections_neg
        ]
        neg_scores = [scores for scores in neg_scores if scores != []]
        im_neg_det_counts = [
            np.sum([np.max(scores) > thres for scores in neg_scores])
            for thres in score_thresholds
        ]
        obj_neg_det_counts = [
            np.sum(np.array(list(itertools.chain(*neg_scores))) > thres)
            for thres in score_thresholds
        ]
        assert (
            len(im_neg_det_counts)
            == len(obj_neg_det_counts)
            == len(score_thresholds)
        )

    else:
        im_neg_det_counts = None
        obj_neg_det_counts = None

    assert (
        len(im_error_counts)
        == len(im_wrong_det_counts)
        == len(im_missed_gt_counts)
        == len(obj_missed_gt_counts)
        == len(obj_wrong_det_counts)
        == len(score_thresholds)
    )

    return (
        score_thresholds,
        im_error_counts,
        im_wrong_det_counts,
        im_missed_gt_counts,
        obj_wrong_det_counts,
        obj_missed_gt_counts,
        im_neg_det_counts,
        obj_neg_det_counts,
    )


class DetectionLearner:
    """ Detection Learner for Object Detection"""

    def __init__(
        self,
        dataset: Dataset = None,
        model: nn.Module = None,
        im_size: int = None,
        device: torch.device = None,
        labels: List[str] = None,
    ):
        """ Initialize leaner object.

        You can only specify an image size `im_size` if `model` is not given.

        Args:
            dataset: the dataset. This class will infer labels if dataset is present.
            model: the nn.Module you wish to use
            im_size: image size for your model
        """
        # if model is None, dataset must not be
        if not model:
            assert dataset is not None

        # not allowed to specify im size if you're providing a model
        if model:
            assert im_size is None

        # if dataset is not None, labels must be (since it is already set in dataset)
        if not dataset:
            assert labels is not None

        # if im_size is not specified, use 500
        if im_size is None:
            im_size = 500

        self.device = device
        if self.device is None:
            self.device = torch_device()

        self.model = model
        self.dataset = dataset
        self.im_size = im_size

        # make sure '__background__' is not included in labels
        if dataset and "labels" in dataset.__dict__:
            self.labels = dataset.labels
        elif labels is not None:
            self.labels = labels
        else:
            raise ValueError("No labels provided in dataset.labels or labels")

        # setup model, default to fasterrcnn
        if self.model is None:
            self.model = get_pretrained_fasterrcnn(
                len(self.labels) + 1,
                min_size=self.im_size,
                max_size=self.im_size,
            )

        self.model.to(self.device)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def fit(
        self,
        epochs: int,
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        print_freq: int = 10,
        step_size: int = None,
        gamma: float = 0.1,
        skip_evaluation: bool = False,
    ) -> None:
        """ The main training loop. """

        if not self.dataset:
            raise Exception("No dataset provided")

        # reduce learning rate every step_size epochs by a factor of gamma (by default) 0.1.
        if step_size is None:
            step_size = int(np.round(epochs / 1.5))

        # construct our optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # store data in these arrays to plot later
        self.losses = []
        self.ap = []
        self.ap_iou_point_5 = []

        # main training loop
        self.epochs = epochs
        for epoch in range(self.epochs):

            # train for one epoch, printing every 10 iterations
            logger = train_one_epoch(
                self.model,
                self.optimizer,
                self.dataset.train_dl,
                self.device,
                epoch,
                print_freq=print_freq,
            )
            
            
            self.losses.append(logger.meters["loss"].median)

            # update the learning rate
            self.lr_scheduler.step()

            # evaluate
            if not skip_evaluation:
                e = self.evaluate(dl=self.dataset.test_dl)
                self.ap.append(_calculate_ap(e))
                self.ap_iou_point_5.append(
                    _calculate_ap(e)
                )

    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss from calling `fit` and average precision on the
        test set. """
        fig = plt.figure(figsize=figsize)
        ap = {k: [dic[k] for dic in self.ap] for k in self.ap[0]}

        for i, (k, v) in enumerate(ap.items()):

            ax1 = fig.add_subplot(1, len(ap), i + 1)

            ax1.set_xlim([0, self.epochs - 1])
            ax1.set_xticks(range(0, self.epochs))
            ax1.set_xlabel("epochs")
            ax1.set_ylabel("loss", color="g")
            ax1.plot(self.losses, "g-")

            ax2 = ax1.twinx()
            ax2.set_ylabel(f"AP for {k}", color="b")
            ax2.plot(v, "b-")

        fig.suptitle("Loss and Average Precision (AP) over Epochs")

    def evaluate(self, dl: DataLoader = None) -> CocoEvaluator:
        """ eval code on validation/test set and saves the evaluation results
        in self.results.

        Raises:
            Exception: if both `dl` and `self.dataset` are None.
        """
        if dl is None:
            if not self.dataset:
                raise Exception("No dataset provided for evaluation")
            dl = self.dataset.test_dl
        self.results = evaluate(self.model, dl, device=self.device)
        return self.results

    def predict(
        self,
        im_or_path: Union[np.ndarray, Union[str, Path]],
        threshold: Optional[int] = 0.5,
    ) -> Dict:
        """ Performs inferencing on an image path or image.

        Args:
            im_or_path: the image array which you can get from
                `Image.open(path)` or a image path
            threshold: the threshold to use to calculate whether the object was
                detected. Note: can be set to None to return all detection
                bounding boxes.

        Return a list of DetectionBbox
        """
        if isinstance(im_or_path, (str, Path)):
            #im = Image.open(im_or_path).convert('RGB')
            im = Image.open(im_or_path)
            im_path = im_or_path
        else:
            im = im_or_path
            im_path = None

        # convert the image to the format required by the model
        transform = transforms.Compose([transforms.ToTensor()])
        im = transform(im)
        if self.device:
            im = im.to(self.device)

        model = self.model.eval()  # eval mode
        with torch.no_grad():
            pred = model([im])[0]

        # detach prediction results to cpu
        pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
        return _extract_od_results(
            _apply_threshold(pred, threshold=threshold), self.labels, im_path
        )

    def predict_dl(
        self, dl: DataLoader, threshold: Optional[float] = 0.5
    ) -> List[DetectionBbox]:
        """ Predict all images in a dataloader object.

        Args:
            dl: the dataloader to predict on
            threshold: iou threshold for a positive detection. Note: set
                threshold to None to omit a threshold

        Returns a list of results
        """
        pred_generator = self.predict_batch(dl, threshold=threshold)
        return [pred for preds in pred_generator for pred in preds]

    def predict_batch(
        self, dl: DataLoader, threshold: Optional[float] = 0.5
    ) -> Generator[List[DetectionBbox], None, None]:
        """ Batch predict

        Args
            dl: A DataLoader to load batches of images from
            threshold: iou threshold for a positive detection. Note: set
                threshold to None to omit a threshold

        Returns an iterator that yields a batch of detection bboxes for each
        image that is scored.
        """

        model = self.model.eval()

        for i, batch in enumerate(dl):
            ims, infos = batch
            ims = [im.to(self.device) for im in ims]
            with torch.no_grad():
                raw_dets = model(ims)

            results = []
            for det, info in zip(raw_dets, infos):
                im_id = int(info["image_id"].item())
                # detach prediction results to cpu
                pred = {k: v.detach().cpu().numpy() for k, v in det.items()}
                extracted_res = _extract_od_results(
                    _apply_threshold(pred, threshold=threshold),
                    self.labels,
                    dl.dataset.dataset.im_paths[im_id],
                )
                results.append({"idx": im_id, **extracted_res})

            yield results

    def save(
        self, name: str, path: str = None, overwrite: bool = True
    ) -> None:
        """ Saves the model

        Save your model in the following format:
        /data_path()
        +-- <name>
        |   +-- meta.json
        |   +-- model.pt

        The meta.json will contain information like the labels and the im_size
        The model.pt will contain the weights of the model

        Args:
            name: the name you wish to save your model under
            path: optional path to save your model to, will use `data_path`
                otherwise
            overwrite: overwrite existing models

        Raise:
            Exception if model file already exists but overwrite is set to
            false

        Returns None
        """
        if path is None:
            path = Path(self.dataset.root) / "models"

        # make dir if not exist
        if not Path(path).exists():
            os.mkdir(path)

        # make dir to contain all model/meta files
        model_path = Path(path) / name
        if model_path.exists():
            if overwrite:
                shutil.rmtree(str(model_path))
            else:
                raise Exception(
                    f"Model of {name} already exists in {path}. Set `overwrite=True` or use another name"
                )
        os.mkdir(model_path)

        # set names
        pt_path = model_path / f"model.pt"
        meta_path = model_path / f"meta.json"

        # save pt
        torch.save(self.model.state_dict(), pt_path)

        # save meta file
        meta_data = {"labels": self.dataset.labels, "im_size": self.im_size}
        with open(meta_path, "w") as meta_file:
            json.dump(meta_data, meta_file)

        print(f"Model is saved to {model_path}")

    def load(self, name: str = None, path: str = None) -> None:
        """ Loads a model.

        Loads a model that is saved in the format that is outputted in the
        `save` function.

        Args:
            name: The name of the model you wish to load. If no name is
            specified, the function will still look for a model under the path
            specified by `data_path`. If multiple models are available in that
            path, it will require you to pass in a name to specify which one to
            use.
            path: Pass in a path if the model is not located in the
            `data_path`. Otherwise it will assume that it is.

        Raise:
            Exception if passed in name/path is invalid and doesn't exist
        """

        # set path
        if not path:
            if self.dataset:
                path = Path(self.dataset.root) / "models"
            else:
                raise Exception("Specify a `path` parameter")

        # if name is given..
        if name:
            model_path = path / name

            pt_path = model_path / "model.pt"
            if not pt_path.exists():
                raise Exception(
                    f"No model file named model.pt exists in {model_path}"
                )

            meta_path = model_path / "meta.json"
            if not meta_path.exists():
                raise Exception(
                    f"No model file named meta.txt exists in {model_path}"
                )

        # if no name is given, we assume there is only one model, otherwise we
        # throw an error
        else:
            models = [f.path for f in os.scandir(path) if f.is_dir()]

            if len(models) == 0:
                raise Exception(f"No model found in {path}.")
            elif len(models) > 1:
                print(
                    f"Multiple models were found in {path}. Please specify which you wish to use in the `name` argument."
                )
                for model in models:
                    print(model)
                exit()
            else:
                pt_path = Path(models[0]) / "model.pt"
                meta_path = Path(models[0]) / "meta.json"

        # load into model
        self.model.load_state_dict(
            torch.load(pt_path, map_location=torch_device())
        )

        # load meta info
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
            self.labels = meta_data["labels"]

    @classmethod
    def from_saved_model(cls, name: str, path: str) -> "DetectionLearner":
        """ Create an instance of the DetectionLearner from a saved model.

        This function expects the format that is outputted in the `save`
        function.

        Args:
            name: the name of the model you wish to load
            path: the path to get your model from

        Returns:
            A DetectionLearner object that can inference.
        """
        path = Path(path)

        meta_path = path / name / "meta.json"
        assert meta_path.exists()

        im_size, labels = None, None
        with open(meta_path) as json_file:
            meta_data = json.load(json_file)
            im_size = meta_data["im_size"]
            labels = meta_data["labels"]

        model = get_pretrained_fasterrcnn(
            len(labels) + 1, min_size=im_size, max_size=im_size
        )
        detection_learner = DetectionLearner(model=model, labels=labels)
        detection_learner.load(name=name, path=path)
        return detection_learner
