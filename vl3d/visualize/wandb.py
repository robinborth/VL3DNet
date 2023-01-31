"""
This module contains helper functions to interact with wandb.
"""

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from vl3d.config import static_cfg

WHITE = [255, 255, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]


def bbox_as_corners(bbox):
    """Converts a bbox to eight points for each corner.

    Args:
        bbox (list): The bbox described with 6 values, thus we have two
        coordinates, a bottom_left corner and a top_right corner.

    Returns:
        list: A bbox with eight corners.
    """
    bbox = np.float64(bbox)
    bottom_left_x, bottom_left_y, bottom_left_z = bbox[:3]
    top_right_x, top_right_y, top_right_z = bbox[3:]
    return [
        [bottom_left_x, bottom_left_y, bottom_left_z],
        [bottom_left_x, top_right_y, bottom_left_z],
        [bottom_left_x, bottom_left_y, top_right_z],
        [top_right_x, bottom_left_y, bottom_left_z],
        [top_right_x, top_right_y, bottom_left_z],
        [bottom_left_x, top_right_y, top_right_z],
        [top_right_x, bottom_left_y, top_right_z],
        [top_right_x, top_right_y, top_right_z],
    ]


def create_wandb_point_cloud(locs, rgb, bboxes, labels, colors):
    """Takes position, normalized rgb color and boxes and creates and wandb object.

    Args:
        locs: The locations of each point.
        rgb: The color of each point normalized between [-1, 1].
        bboxes: The bboxes that should be displayed in the scene.
        lables: The labels for each bbox that is displayed.
        colors: The color for each bbox.

    Returns:
        wandb.Object3D: A object that contains the points and colors in the
        scene. In the scene are the bounding boxes.
    """
    unnormalized_rgb = (rgb + 1.0) * 127.5
    points = np.concatenate((locs, unnormalized_rgb), axis=1)

    boxes = []
    for bbox, label, color in zip(bboxes, labels, colors):
        boxes.append({"color": color, "label": label, "corners": bbox_as_corners(bbox)})
    boxes = np.array(boxes)

    return wandb.Object3D({"type": "lidar/beta", "points": points, "boxes": boxes})


def topk_bboxes(pred_bboxes, pred_bboxes_logits, pred_sem_label, k: int = 3, largest: bool = True):
    """Return the best or worst k bboxes given confidence scores"""
    topk_idxs = torch.squeeze(torch.topk(pred_bboxes_logits, dim=1, k=k, largest=largest).indices)
    bboxes, bbox_score, bbox_label, bbox_color = [], [], [], []
    for idx in range(topk_idxs.size(0)):
        bboxes.append(pred_bboxes[idx, topk_idxs[idx]])

        pred_bbox_score = F.softmax(pred_bboxes_logits, dim=1)
        bbox_score.append(pred_bbox_score[idx, topk_idxs[idx]])

        labels, colors = [], []
        for i in range(k):
            cls_name = static_cfg.scannetv2.class_names[int(pred_sem_label[idx])]
            if i == 0:
                labels.append(f"PRED(1)\ncls: {cls_name}\nscore: {bbox_score[idx][i]:.3f}")
                colors.append(RED)
            else:
                labels.append(f"TOP({i + 1})\nscore: {bbox_score[idx][i]:.3f}")
                colors.append(GREEN)
        bbox_label.append(labels), bbox_color.append(colors)

    return {
        "topk_pred_bboxes": torch.stack(bboxes),
        "topk_pred_bbox_score": torch.stack(bbox_score),
        "topk_pred_bbox_label": bbox_label,
        "topk_pred_bbox_color": bbox_color,
    }


def create_wandb_grounding_scene(output_dict, idx: int = 0):
    gt_bbox = output_dict["gt_bbox"].cpu().numpy()[idx]
    topk = topk_bboxes(
        pred_bboxes=output_dict["pred_bboxes"],
        pred_bboxes_logits=output_dict["pred_bboxes_logits"],
        pred_sem_label=output_dict["pred_sem_label"],
    )

    locs = output_dict["locs"][idx].cpu().numpy()
    rgb = output_dict["rgb"][idx].cpu().numpy()
    topk_pred_bboxes = topk["topk_pred_bboxes"].cpu().numpy()[idx]
    topk_pred_bbox_label = topk["topk_pred_bbox_label"][idx]
    topk_pred_bbox_color = topk["topk_pred_bbox_color"][idx]

    point_cloud = create_wandb_point_cloud(
        locs=locs,
        rgb=rgb,
        bboxes=[gt_bbox, *topk_pred_bboxes],
        labels=["", *topk_pred_bbox_label],
        colors=[WHITE, *topk_pred_bbox_color],
    )

    return point_cloud
