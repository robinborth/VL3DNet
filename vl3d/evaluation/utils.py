"""
This module contains utilities to helps to evaluate the experiments.
"""

import numpy as np
import torch
import torch.nn.functional as F


def bbox_iou(bboxes_a, bboxes_b):
    """
    Get the pointwise intersection over union for two lists of bboxes.

    Args:
        bboxes_a, bboxes_b: Tensor of shape (num_boxes, 6)

    Returns:
        a tensor of shape (num_boxes, ) from 0-1 indicating the intersection over union value.
        0 indicates non overlapping.
    """
    bottom_left_a, top_right_a = bboxes_a[:, :3], bboxes_a[:, 3:]
    bottom_left_b, top_right_b = bboxes_b[:, :3], bboxes_b[:, 3:]

    min_max = torch.min(torch.stack((top_right_a, top_right_b), dim=1), dim=1).values
    max_min = torch.max(torch.stack((bottom_left_a, bottom_left_b), dim=1), dim=1).values

    intersection = torch.prod(F.relu(min_max - max_min), dim=1)

    volume_a = torch.prod(top_right_a - bottom_left_a, dim=1)
    volume_b = torch.prod(top_right_b - bottom_left_b, dim=1)

    union = volume_a + volume_b - intersection
    return 1.0 * intersection / union


def bbox_iou_numpy(bboxes_a, bboxes_b):
    """
    Get the pointwise intersection over union for two lists of bboxes.

    Args:
        bboxes_a, bboxes_b: Numpy array of shape (num_boxes, 6)

    Returns:
        a numpy array of shape (num_boxes, ) from 0-1 indicating the intersection over union value.
        0 indicates non overlapping.
    """
    bottom_left_a, top_right_a = bboxes_a[:, :3], bboxes_a[:, 3:]
    bottom_left_b, top_right_b = bboxes_b[:, :3], bboxes_b[:, 3:]

    min_max = np.fmin(top_right_a, top_right_b)
    max_min = np.fmax(bottom_left_a, bottom_left_b)

    intersection = np.prod(np.maximum(min_max - max_min, 0), axis=1)

    volume_a = np.prod(top_right_a - bottom_left_a, axis=1)
    volume_b = np.prod(top_right_b - bottom_left_b, axis=1)

    union = volume_a + volume_b - intersection
    return 1.0 * intersection / union
