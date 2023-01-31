"""
The collate functions for the different datasets e.g. scannetv2 and scanrefer.
The collate functions justs takes a batch, containing of different dataset[i]
items and group them together under a dictionary. So when working with batches
someone can access dataset[i] content by same keys then dataset[i] and the
index of the item in the batch.
"""

from typing import Any

import torch

# HACK current version don't use minsu3d
try:
    from minsu3d.common_ops.functions import common_ops
except ImportError as e:
    pass


def softgroup_collate_fn(batch):
    """The dataset collate function for preprocessed scene data"""
    data = {}

    scene_id = []
    locs = []
    rgb = []

    instance_id = []
    unique_mask = []
    sem_label = []
    ann_id = []

    gt_bbox = []
    pred_bboxes = []
    pred_bboxes_features = []
    pred_bboxes_attention_mask = []
    pred_bboxes_labels = []
    pred_bboxes_conf = []

    input_ids = []
    langugage_attention_mask = []

    for b in batch:
        scene_id.append(b["scene_id"])
        locs.append(b["locs"])
        rgb.append(b["rgb"])

        instance_id.append(b["instance_id"])
        unique_mask.append(b["unique_mask"])
        sem_label.append(b["gt_sem_label"])
        ann_id.append(b["ann_id"])

        gt_bbox.append(b["gt_bbox"])
        pred_bboxes.append(b["pred_bboxes"])
        pred_bboxes_features.append(b["pred_bboxes_features"])
        pred_bboxes_attention_mask.append(b["pred_bboxes_attention_mask"])
        pred_bboxes_labels.append(b["pred_bboxes_labels"])
        pred_bboxes_conf.append(b["pred_bboxes_conf"])

        input_ids.append(b["input_ids"])
        langugage_attention_mask.append(b["language_attention_mask"])

    data["scene_id"] = scene_id
    data["locs"] = locs
    data["rgb"] = rgb

    data["instance_id"] = torch.stack(instance_id, dim=0)
    data["unique_mask"] = torch.stack(unique_mask, dim=0)
    data["gt_sem_label"] = torch.stack(sem_label, dim=0)
    data["ann_id"] = torch.stack(ann_id, dim=0)

    data["gt_bbox"] = torch.stack(gt_bbox, dim=0)
    data["pred_bboxes"] = torch.stack(pred_bboxes, dim=0)
    data["pred_bboxes_features"] = torch.stack(pred_bboxes_features, dim=0)
    data["pred_bboxes_attention_mask"] = torch.stack(pred_bboxes_attention_mask, dim=0)
    data["pred_bboxes_labels"] = torch.stack(pred_bboxes_labels, dim=0)
    data["pred_bboxes_conf"] = torch.stack(pred_bboxes_conf, dim=0)

    data["input_ids"] = torch.stack(input_ids, dim=0)
    data["language_attention_mask"] = torch.stack(langugage_attention_mask, dim=0)

    return data


def scannetv2_collate_fn(batch: list) -> dict:
    """The main collate function.

    This collate funciton combines the different function
    and makes it easy to enable and disable different functions.
    This function is used in the SceneDataModule.

    Args:
        batch (list): A list containing multiple items of the dataset.

    Returns:
        dict: A dict that allows to access the value of the items in the batch
        with the same key as you would do when just accessing the item directly.
    """
    data: dict[str, Any] = {}
    data.update(meta_collate_fn(batch))
    data.update(scanrefer_collate_fn(batch))
    data.update(scannetv2_collate_fn(batch))
    return data


def meta_collate_fn(batch: list) -> dict:
    """Combines the meta information over the items in the SceneDataset."""
    data: dict[str, Any] = {}

    scene_ids = []
    chunk_ids = []
    chunk_sizes = []
    actual_chunk_sizes = []

    for b in batch:
        scene_ids.append(b["scene_id"])
        chunk_ids.append(b["chunk_id"])
        chunk_sizes.append(b["chunk_size"])
        actual_chunk_sizes.append(b["actual_chunk_size"])

    data["scene_ids"] = scene_ids
    data["chunk_ids"] = torch.tensor(chunk_ids, dtype=torch.int32)
    data["chunk_sizes"] = torch.tensor(chunk_sizes, dtype=torch.int32)
    data["actual_chunk_sizes"] = torch.tensor(actual_chunk_sizes, dtype=torch.int32)

    return data


def scanrefer_collate_fn(batch: list) -> dict:
    """Combines the scene object information over the items in the SceneDataset."""
    data: dict[str, Any] = {}

    instance_id = []
    unique_mask = []
    sem_label = []
    ann_id = []
    gt_bbox = []
    input_ids = []
    token_type_ids = []
    attention_mask = []

    for b in batch:
        instance_id.extend(b["instance_id"])
        unique_mask.extend(b["unique_mask"])
        sem_label.extend(b["sem_label"])
        ann_id.extend(b["ann_id"])
        gt_bbox.append(torch.from_numpy(b["gt_bbox"]))
        input_ids.append(b["input_ids"])
        token_type_ids.append(b["token_type_ids"])
        attention_mask.append(b["attention_mask"])

    data["instance_id"] = torch.tensor(instance_id, dtype=torch.int32)
    data["unique_mask"] = torch.tensor(unique_mask, dtype=torch.int32)
    data["sem_label"] = torch.tensor(sem_label, dtype=torch.int32)
    data["ann_id"] = torch.tensor(sem_label, dtype=torch.int32)
    data["gt_bbox"] = torch.cat(gt_bbox, dim=0)
    data["input_ids"] = torch.cat(input_ids, dim=0).type(torch.int64)
    data["token_type_ids"] = torch.cat(token_type_ids, dim=0)
    data["attention_mask"] = torch.cat(attention_mask, dim=0)

    return data


def scannetv2_collate_fn(batch: list) -> dict:
    """Combines the scene information over the items in the SceneDataset.

    This collate function is similar to sparse_collate_fn from minsu3d. However
    there are some minor changes. First we dropped scan_id here, we add that in
    meta_collate_fn() and renamed it to scene_id. We also added the rgb values
    for each point and a mapping from instance_ids to instance_idx. Finally we
    renamed instance_label_cls to instance_sem_label.
    """
    data: dict[str, Any] = {}

    locs = []
    rgb = []
    locs_scaled = []
    feats = []
    vert_batch_ids = []
    sem_labels = []

    instance_ids = []
    instance_info = []
    instance_num_point = []
    instance_sem_label = []
    instance_id2instance_idx = []
    instance_offsets = [0]
    total_num_inst = 0

    for i, b in enumerate(batch):
        rgb.append(torch.from_numpy(b["rgb"]))
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16))
        feats.append(torch.from_numpy(b["feats"]))

        instance_ids_i = b["instance_ids"]
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_sem_label.extend(b["instance_sem_label"])
        instance_id2instance_idx.extend(b["instance_id2instance_idx"])

    data["locs"] = torch.cat(locs, dim=0)
    data["rgb"] = torch.cat(rgb, dim=0)
    data["feats"] = torch.cat(feats, dim=0)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["sem_labels"] = torch.cat(sem_labels, dim=0)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)

    data["instance_info"] = torch.cat(instance_info, dim=0)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)
    data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int32)
    data["instance_sem_label"] = torch.tensor(instance_sem_label, dtype=torch.int32)
    data["instance_id2instance_idx"] = torch.tensor(instance_id2instance_idx, dtype=torch.int32)

    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(
        torch.cat(locs_scaled, dim=0), data["vert_batch_ids"], len(batch), 4
    )

    return data
