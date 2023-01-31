"""
This module defines utility functions, to extract and preprocess the scene data.
"""
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

from vl3d.config import static_cfg
from vl3d.evaluation.utils import bbox_iou, bbox_iou_numpy


def get_scene_info(
    scene: dict,
    scale: int,
    ignore_label: int,
    ignore_classes: list[str],
    class_names,
) -> dict:
    """Extract scene information for scannetv2.

    Args:
        scene (dict): The scene as point cloud. It needs to contain the
        information for: "xyz", "rgb", "instance_ids" and "sem_labels".
        scale (int): The ammount of scaling for each point xyz.
        ignore_label (int): The instance_ids that are ignored.

    Returns:
        dict: A dictionary that contains scannetv2 information, that is
        needed for further processing e.g. sparse_collate_fn().
    """
    data = {}

    data["locs"] = scene["xyz"]  # (N, 3)
    data["rgb"] = scene["rgb"]  # (N, 3)

    scaled_points = data["locs"] * scale
    scaled_points -= scaled_points.min(axis=0)
    data["locs_scaled"] = scaled_points  # (N, 3)

    feats = np.zeros(shape=(len(scaled_points), 0), dtype=np.float32)
    feats = np.concatenate((feats, data["rgb"]), axis=1)
    data["feats"] = feats  # (N, 3)

    data["sem_labels"] = sem_labels_without_ignore_classes(
        sem_labels=scene["sem_labels"],
        class_names=class_names,
        ignore_classes=ignore_classes,
    )  # (N,)
    data["instance_ids"] = scene["instance_ids"]  # (N,) 0~total_nInst, -1

    # The number of instance in the scene
    unique_instance_ids = np.unique(data["instance_ids"])
    unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label]
    num_instance = unique_instance_ids.shape[0]
    data["num_instance"] = np.array(num_instance, dtype=np.int32)  # (1, )

    # The center point of each instance
    instance_info = np.empty(shape=(data["locs"].shape[0], 3), dtype=np.float32)
    for index, instance_id in enumerate(unique_instance_ids):
        idx = data["instance_ids"] == instance_id
        xyz_i = data["locs"][idx]
        mean_xyz_i = xyz_i.mean(0)
        instance_info[index] = mean_xyz_i
    data["instance_info"] = instance_info  # (num_instance, 3)

    # The number of points for each instance
    num_points = []
    for instance_id in unique_instance_ids:
        idx = data["instance_ids"] == instance_id
        num_points.append(idx.sum())
    data["instance_num_point"] = np.array(num_points, dtype=np.int32)  # (num_instance,)

    # The semantic class for each instance
    instance_cls = np.full(shape=num_instance, fill_value=ignore_label, dtype=np.int8)
    for index, instance_id in enumerate(unique_instance_ids):
        idx = data["instance_ids"] == instance_id
        instance_sem_labels = np.unique(data["sem_labels"][idx])
        assert len(instance_sem_labels) == 1
        instance_cls[index] = instance_sem_labels[0]
    data["instance_sem_label"] = instance_cls  # (num_instance, )

    # The mapping from instance_id to the 'instance_*' information
    instance_id2idx_size = data["instance_ids"].max() + 1
    instance_id2idx = np.full(shape=instance_id2idx_size, fill_value=ignore_label, dtype=np.int8)
    for index, instance_id in enumerate(unique_instance_ids):
        instance_id2idx[instance_id] = index
    data["instance_id2instance_idx"] = instance_id2idx

    return data


def get_unique_mask_info(sem_labels, scene_objects):
    unique_mask = np.zeros(len(sem_labels))
    for idx, object_id in enumerate(sem_labels):
        hits = 0
        for scene_object in scene_objects:
            if scene_object["object_id"] == object_id:
                hits += 1
        unique_mask[idx] = min(hits, 1)
    return {"unique_mask": unique_mask}


def get_object_name2sem_label(
    scannetv2_labels_path: str | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Creates a mapping dict from object_names to class_names.

    Args:
        scannetv2_labels (pd.DataFrame): The scannetv2-labels.combined.tsv dataset loaded as
        pandas dataframe.
        class_names (list[str]): The list of class names that are used in the visual grounding
        and dense captioning task.

    Returns:
        str: A key, value dictionary where the key is the object_name which
        you can get from the ScanRefer_filtered.json dataset and value is
        the class_name that is used for the tasks.
    """
    if scannetv2_labels_path is None:
        scannetv2_labels_path = static_cfg.scannetv2.labels_combined_path
    if class_names is None:
        class_names = static_cfg.scannetv2.class_names

    scannetv2_labels = pd.read_csv(scannetv2_labels_path, sep="\t")
    object_names = scannetv2_labels["raw_category"].str.replace(" ", "_")
    nyu40classes = scannetv2_labels["nyu40class"]
    obj2sem = {}
    for object_name, nyu40class in zip(object_names, nyu40classes):
        if nyu40class not in class_names:
            obj2sem[object_name] = class_names.index("others")
        else:
            obj2sem[object_name] = class_names.index(nyu40class)
    return obj2sem


def sem_labels_without_ignore_classes(sem_labels, class_names, ignore_classes):
    """Convert sem_labels to class_names_idx

    Args:
        sem_labels (np.array): The array of sem_labels from scannetv2.

    Returns:
        np.array: An array of indexes for the class_names.
    """
    map_to_other_idx = (sem_labels == 0) | (sem_labels == 1)
    reduce_by_ignore_classes_idx = sem_labels > 1
    sem_labels[map_to_other_idx] = class_names.index("others")
    sem_labels[reduce_by_ignore_classes_idx] -= len(ignore_classes)
    return sem_labels


def clip_tensors(tensor, max_length, fill_value=0):
    """Padds and truncate 2d tensor.

    Args:
        tensor: A 2d tensor of shape (x, y)
        max_length: The dimension of x after clipping
        fill_value (int, optional): The value of the clipped tensors. Defaults to 0.

    Returns:
        A tensor of shape (max_length, y)
    """
    tensor = tensor[:max_length]  # (min(len(tensor), max_length), y)
    tensor_size, feature_size = tensor.shape
    pad_size = max_length - tensor_size
    zeros = tensor.new_full((pad_size, feature_size), fill_value=fill_value)
    return torch.cat((tensor, zeros), dim=0)  # (max_length, y)


def get_scene_object_info(
    chunk: dict,
    chunk_size: int,
    actual_chunk_size: int,
    tokenizer: BertTokenizer,
    max_length: int,
) -> dict:
    """Extract the information from the scanrefer datset.

    Args:
        chunk (dict): A scene with multiple objects.
        chunk_size (int): The number of objects in one chunk.
        actual_chunk_size (int): The real number of objects in one chunk.
        tokenizer: The bert tokenizer to process the descriptions.
        max_length: The size of the tokens after tokenizing. We apply truncuation
        and padding to the max_length to ensure that all input sequences have the
        same length.

    Returns:
        dict: A dictionary that contains scanrefer information, that is
        needed for further processing e.g. align gt_bbox with the object.
    """
    data: dict[str, Any] = {}

    instance_id = np.full(shape=chunk_size, fill_value=-1, dtype=np.int8)
    sem_label = np.full(shape=chunk_size, fill_value=-1, dtype=np.int8)
    ann_id = np.full(shape=chunk_size, fill_value=-1, dtype=np.int8)
    unique_mask = np.full(shape=chunk_size, fill_value=-1, dtype=np.int8)
    gt_bbox = np.full(shape=(chunk_size, 6), fill_value=0, dtype=np.float32)

    all_gt_bbox_list = get_gt_bbox(chunk["xyz"], chunk["instance_ids"])
    all_gt_bbox_dict = {instance_id: bbox for instance_id, bbox in all_gt_bbox_list}

    for idx in range(actual_chunk_size):
        instance_id[idx] = chunk["object_id"][idx]
        sem_label[idx] = chunk["sem_label"][idx]
        ann_id[idx] = chunk["ann_id"][idx]
        unique_mask[idx] = chunk["unique_mask"][idx]
        gt_bbox[idx] = all_gt_bbox_dict[instance_id[idx]]

    data["instance_id"] = instance_id
    data["sem_label"] = sem_label
    data["ann_id"] = ann_id
    data["unique_mask"] = unique_mask
    data["gt_bbox"] = gt_bbox

    # Because descriptions is allready a list, we can direktly use the
    # tokenizer from huggingface. However if we are in the last chunk of a
    # scene this produces an encoded_input of shape (actual_chunk_size, max_length).
    # We pad this case with zeroes to obtain a shape (chunk_size, max_length).
    encoded_input = tokenizer(
        text=chunk["description"],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    for key, value in encoded_input.items():
        data[key] = clip_tensors(value, max_length=chunk_size, fill_value=0)

    return data


def enrich_scene_objects(
    scenes_objects,
    scenes,
    proposals_max_count,
    use_euclidean_backup,
    drop_sample_iou_threshold,
):
    """Adds meta information to each object in the scene.

    Args:
        scene_objects: A dictionary where each value is a list of object in the scene.

    Returns:
        For each object in each scene add the "sem_label" and an "unique_mask".
    """
    object_name2sem_label = get_object_name2sem_label()

    for scene_id, scene in scenes_objects.items():
        counter = Counter({s["object_id"]: object_name2sem_label[s["object_name"]] for s in scene}.values())
        sem_label2unique_mask = {sem_label: min(hits - 1, 1) for sem_label, hits in counter.items()}
        all_gt_bboxes = {instance_id: bbox for instance_id, bbox in scenes[scene_id]["gt_bboxes"]}
        pred_bboxes = scenes[scene_id]["pred_bboxes"][:proposals_max_count]
        for idx, so in enumerate(scene):
            so["instance_id"] = int(so["object_id"])
            so["sem_label"] = object_name2sem_label[so["object_name"]]
            so["ann_id"] = int(so["ann_id"])
            so["unique_mask"] = sem_label2unique_mask[so["sem_label"]]
            so["gt_bbox"] = all_gt_bboxes[so["instance_id"]]

            gt_pred_bbox_idx, pred_bbox_found = extract_ids_for_pred_bboxes(
                pred_bboxes=pred_bboxes,
                gt_bbox=so["gt_bbox"],
                use_euclidean_backup=use_euclidean_backup,
                drop_sample_iou_threshold=drop_sample_iou_threshold,
            )
            so["gt_pred_bbox_idx"] = int(gt_pred_bbox_idx)
            so["pred_bbox_found"] = bool(pred_bbox_found)

            scenes_objects[scene_id][idx] = so

    return scenes_objects


def extract_ids_for_pred_bboxes(
    pred_bboxes,
    gt_bbox,
    use_euclidean_backup=True,
    drop_sample_iou_threshold=0.0,
):
    """Extracts coresponding pred_bbox id for the corresponding gt_bbox.

    Args:
        pred_bboxes: The predicted bboxes of shape (vision_max_length, 6)
        gt_bbox: The ground truth bbox for the instance of size (6,)

    Returns:
        A numpy array of shape (1, ), and a bool of wheter we found
    """
    num_pred_bbox = pred_bboxes.shape[0]
    gt_bbox_repeated = np.repeat(gt_bbox.reshape(1, -1), num_pred_bbox, axis=0)
    pred_ious = bbox_iou_numpy(gt_bbox_repeated, pred_bboxes)

    # when there is an overlap between prediction and gt
    pred_bbox_found = (pred_ious >= drop_sample_iou_threshold).sum()
    if pred_bbox_found or not use_euclidean_backup:
        return np.argmax(pred_ious), pred_bbox_found

    # else need to use the distance between two objects
    pred_bboxes_centers = np.mean(pred_bboxes.reshape(-1, 2, 3), axis=1)
    gt_bbox_center = np.mean(gt_bbox.reshape(2, 3), axis=0)
    pred_dist = np.squeeze(np.linalg.norm(pred_bboxes_centers - gt_bbox_center, axis=1))
    return np.argmin(pred_dist), pred_bbox_found


def get_gt_bbox(xyz, instance_ids, ignored_label=-1):
    """
    Get all ground truth bounding boxes for one scene, together with the
    instance_id that belongs to that ground truth bounding box. (The instance_id
    is the same as the object_id in ScanRefer.)

    Args:
        xyz: The points in the pointcloud, this can also be called "locs".
        instance_ids: The id of the instance that the point belongs to.
        ignored_label: Points with this label are not considered.

    Returns:
        A list of tuples. The first element is the instance_id, the second
        is a list of positions that markes the bounding box. The box consists of
        6 positions.
    """
    gt_bbox = []
    unique_inst_ids = np.unique(instance_ids)
    for instance_id in unique_inst_ids:
        if instance_id == ignored_label:
            continue
        idx = instance_ids == instance_id
        xyz_i = xyz[idx]
        min_xyz = xyz_i.min(0)
        max_xyz = xyz_i.max(0)
        gt_bbox.append((instance_id, np.concatenate((min_xyz, max_xyz))))
    return gt_bbox


def get_nms_instances(pred_bboxes, scores, threshold):
    """Non max suppression for 3D instance proposals based on cross ious and scores.

    Returns:
        The indexes that should be keeped for the pred_bboxes.
    """
    ious = []
    for pred_bbox in pred_bboxes:
        stacked = torch.cat([pred_bbox.unsqueeze(0) for i in range(len(pred_bboxes))])
        iou = bbox_iou(stacked, pred_bboxes)
        ious.append(iou)
    cross_ious = torch.stack(ious).cpu().numpy()
    ixs = np.argsort(-scores.cpu().numpy())  # descending order
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        ious = cross_ious[i, ixs[1:]]
        remove_ixs = np.where(ious > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return torch.tensor(pick, dtype=torch.int32).to(device=scores.device)
