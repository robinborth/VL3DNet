"""
This module consists of the SceneDataset which combines the scannetv2 dataset
and the scanrefer dataset into one unified dataset.
"""
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from vl3d.config import static_cfg
from vl3d.data.utils import (
    enrich_scene_objects,
    extract_ids_for_pred_bboxes,
    get_object_name2sem_label,
    get_scene_info,
    get_scene_object_info,
)


class SoftgroupSceneDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        token_max_count: int = 60,
        proposal_max_count: int = 60,
        use_bbox: bool = True,
        use_euclidean_backup: bool = True,
        drop_sample_iou_threshold: float = 0.0,
    ) -> None:
        self.split = split
        self.token_max_count = token_max_count
        self.proposal_max_count = proposal_max_count
        self.use_bbox = use_bbox
        self.use_euclidean_backup = use_euclidean_backup
        self.drop_sample_iou_threshold = drop_sample_iou_threshold
        self._load()

    def _load_scanrefer(self):
        """Loads the scanrefer dataset.

        Returns:
            A dict, wehre each key is the name of one scene. The value is a list
            of scene objects in that scene.
        """
        with open(static_cfg.scanrefer[f"{self.split}_split"], "r") as f:
            scenes_objects_raw = json.load(f)
        scene_names = list(set([scene["scene_id"] for scene in scenes_objects_raw]))

        # Transform the list of scene objects to the dict
        scenes_objects = defaultdict(list)
        for scene_object in scenes_objects_raw:
            scene_id = scene_object["scene_id"]
            scenes_objects[scene_id].append(scene_object)

        return scenes_objects, scene_names

    def _load_scannetv2(self):
        """Loads all scenes as dict.

        Returns:
            A dict, where the keys are scene_names e.g. scene0000_00 and the
            values are also dicts with information about the scene e.g. "xyz"
            info.
        """
        scenes = defaultdict(dict)

        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data..."):
            path = os.path.join(static_cfg.softgroup.path, f"{self.split}", f"{scene_name}.pth")
            scene = torch.load(path)
            scene["locs"] -= scene["locs"].mean(axis=0)
            scene["rgb"] = scene["rgb"]
            scenes[scene_name] = scene
        return scenes

    def _load(self) -> None:
        """Load all the data into the dataset."""
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        scenes_objects_raw, self.scene_names = self._load_scanrefer()
        self.scenes = self._load_scannetv2()

        scenes_objects_enritched = enrich_scene_objects(
            scenes_objects=scenes_objects_raw,
            scenes=self.scenes,
            proposals_max_count=self.proposal_max_count,
            use_euclidean_backup=self.use_euclidean_backup,
            drop_sample_iou_threshold=self.drop_sample_iou_threshold,
        )
        self.scenes_objects = []
        for scene_object_per_scene in scenes_objects_enritched.values():
            for scene_object in scene_object_per_scene:
                if scene_object["pred_bbox_found"]:
                    self.scenes_objects.append(scene_object)

    def __len__(self) -> int:
        """The size of the chunked scanrefer dataset."""
        return len(self.scenes_objects)

    def __getitem__(self, idx: int):
        """A scene with descriptions to the objects and meta information."""
        data = {}

        # Meta Information
        scene_id = self.scenes_objects[idx]["scene_id"]

        data["scene_id"] = scene_id
        data["locs"] = torch.from_numpy(self.scenes[scene_id]["locs"])
        data["rgb"] = torch.from_numpy(self.scenes[scene_id]["rgb"])

        data["instance_id"] = torch.tensor(self.scenes_objects[idx]["instance_id"], dtype=torch.int32)
        data["gt_sem_label"] = torch.tensor(self.scenes_objects[idx]["sem_label"], dtype=torch.int32)
        data["unique_mask"] = torch.tensor(self.scenes_objects[idx]["unique_mask"], dtype=torch.int32)
        data["ann_id"] = torch.tensor(self.scenes_objects[idx]["ann_id"], dtype=torch.int32)

        data["gt_bbox"] = torch.from_numpy(self.scenes_objects[idx]["gt_bbox"])

        # Proposal Features
        size = self.proposal_max_count

        pred_bboxes_truncated = torch.from_numpy(self.scenes[scene_id]["pred_bboxes"])[:size]
        mappings = torch.randperm(size)[: len(pred_bboxes_truncated)]
        pred_bboxes = pred_bboxes_truncated.new_full((size, 6), fill_value=0)
        pred_bboxes[mappings] = pred_bboxes_truncated
        data["pred_bboxes"] = pred_bboxes

        proposals_features_truncated = torch.from_numpy(self.scenes[scene_id]["proposals_features"])[:size]
        if self.use_bbox:
            proposals_features_truncated = torch.concat((proposals_features_truncated, pred_bboxes_truncated), dim=1)
        proposals_feature_size = 32 + self.use_bbox * 6
        proposals_features = proposals_features_truncated.new_full((size, proposals_feature_size), fill_value=0)
        proposals_features[mappings] = proposals_features_truncated
        data["pred_bboxes_features"] = proposals_features

        pred_bboxes_attention_mask = pred_bboxes_truncated.new_full((size,), fill_value=1)
        pred_bboxes_attention_mask[mappings] = 0
        data["pred_bboxes_attention_mask"] = pred_bboxes_attention_mask

        gt_pred_bbox_idx = self.scenes_objects[idx]["gt_pred_bbox_idx"]
        pred_bboxes_labels = pred_bboxes.new_full((size,), fill_value=0)
        pred_bboxes_labels[mappings[gt_pred_bbox_idx]] = 1
        data["pred_bboxes_labels"] = pred_bboxes_labels

        pred_bboxes_conf_truncated = torch.from_numpy(self.scenes[scene_id]["pred_bboxes_conf"])[:size]
        pred_bboxes_conf = pred_bboxes_conf_truncated.new_full((size,), fill_value=0)
        pred_bboxes_conf[mappings] = pred_bboxes_conf_truncated
        data["pred_bboxes_conf"] = pred_bboxes_conf

        # Language Features
        encoded_input = self.tokenizer(
            text=self.scenes_objects[idx]["description"],
            return_tensors="pt",
            max_length=self.token_max_count,
            truncation=True,
            padding="max_length",
        )
        data["input_ids"] = encoded_input["input_ids"][0].type(torch.int64)
        data["language_attention_mask"] = encoded_input["attention_mask"][0]

        return data


class ScannetV2SceneDataset(Dataset):
    def __init__(self, split: str, debug: bool = False) -> None:
        self.split = split
        self.debug = debug
        self._load()

    def _load_scanrefer(self):
        """Loads the scanrefer dataset.

        Returns:
            A dict, wehre each key is the name of one scene. The value is a list
            of scene objects in that scene.
        """
        with open(static_cfg.scanrefer[f"{self.split}_split"], "r") as f:
            scenes_objects_raw = json.load(f)

        # When degug_scenes, keep the objs that are in the {split}_scenes list
        if self.debug:
            scenes = static_cfg.data[f"{self.split}_scenes"]
            scenes_objects_raw = [obj for obj in scenes_objects_raw if obj["scene_id"] in scenes]

        # Transform the list of scene objects to the dict
        scenes_objects = defaultdict(list)
        for scene_object in scenes_objects_raw:
            scene_id = scene_object["scene_id"]
            scenes_objects[scene_id].append(scene_object)
        return scenes_objects

    def _load_scannetv2(self):
        """Loads all scenes as dict.

        Returns:
            A dict, where the keys are scene_names e.g. scene0000_00 and the
            values are also dicts with information about the scene e.g. "xyz"
            info.
        """
        scenes = defaultdict(dict)
        for scene_name in tqdm(self.scenes_objects, desc=f"Loading {self.split} data..."):
            path = os.path.join(static_cfg.scannetv2.path, self.split, f"{scene_name}.pth")
            scene = torch.load(path)
            scene["xyz"] -= scene["xyz"].mean(axis=0)
            scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            scenes[scene_name] = scene
        return scenes

    def _get_chunked_data(self, chunk_size: int) -> list[dict]:
        """Chunks the scene dataset and add object information to scene.

        One element in the chunked_data contains of one scene e.g. scene_0000_00
        together with all the scannetv2 information with that scene. And object
        informations for chunk_size objects in that scene. We do that because it
        can happen that in one scene are x > chunk_size objects. We want to split
        these objects into multiple items in the dataset, for performance reasons.

        Args:
            chunk_size (int): The number of objects per chunk.

        Returns:
            list[dict]: Returns a list containing all scenes with all objects.
        """
        chunked_data = []
        for scene_id in self.scenes:
            scene_objects = self.scenes_objects[scene_id]
            for chunk in self._chunks(scene_objects, chunk_size):
                scene = deepcopy(self.scenes[scene_id])
                tmp_dict = defaultdict(list)
                for scene_object in chunk:
                    for key, value in scene_object.items():
                        tmp_dict[key].append(value)
                scene.update(tmp_dict)
                chunked_data.append(scene)
        return chunked_data

    def _chunks(self, lst, n):
        """Splits a list into smaller lists with size n."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _enrich_scene_objects(self, scenes_objects):
        """Adds meta information to each object in the scene.

        Args:
            scene_objects: A dictionary where each value is a list of object in the scene.

        Returns:
            For each object in each scene add the "sem_label" and an "unique_mask".
        """
        scannetv2_labels = pd.read_csv(static_cfg.scannetv2.labels_combined_path, sep="\t")
        object_name2sem_label = get_object_name2sem_label(
            scannetv2_labels=scannetv2_labels,
            class_names=static_cfg.scannetv2.class_names,
        )

        for scene_id, scene in scenes_objects.items():
            for idx, scene_object in enumerate(scene):
                sem_label = object_name2sem_label[scene_object["object_name"]]
                scene_object["sem_label"] = sem_label
                scenes_objects[scene_id][idx] = scene_object

        for scene_id, scene in scenes_objects.items():
            object_id2sem_label = {s["object_id"]: s["sem_label"] for s in scene}
            counter = Counter(object_id2sem_label.values())
            sem_label2unique_mask = {}
            for sem_label, hits in counter.items():
                hits -= 1
                sem_label2unique_mask[sem_label] = min(hits, 1)
            for idx, scene_object in enumerate(scene):
                scene_object["unique_mask"] = sem_label2unique_mask[scene_object["sem_label"]]
                scenes_objects[scene_id][idx] = scene_object

        return scenes_objects

    def _load(self) -> None:
        """Load all the data into the dataset."""
        scenes_objects = self._load_scanrefer()
        self.scenes_objects = self._enrich_scene_objects(scenes_objects)
        self.scenes = self._load_scannetv2()

        self.chunk_size = 30
        self.chunked_data = self._get_chunked_data(self.chunk_size)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self) -> int:
        """The size of the chunked scanrefer dataset."""
        return len(self.chunked_data)

    def __getitem__(self, idx: int):
        """A scene with descriptions to the objects and meta information."""
        data = {}

        data["chunk_id"] = idx
        data["scene_id"] = self.chunked_data[idx]["scene_id"][0]
        data["chunk_size"] = self.chunk_size
        data["actual_chunk_size"] = len(self.chunked_data[idx]["scene_id"])

        # Adds all information to the scene.
        scene_data = get_scene_info(
            scene=self.chunked_data[idx],
            scale=static_cfg.scannetv2.scale,
            ignore_label=static_cfg.scannetv2.ignore_label,
            ignore_classes=static_cfg.scannetv2.ignore_classes,
            class_names=static_cfg.scannetv2.class_names,
        )
        data.update(scene_data)

        # Adds the information for the objects in the scene.
        scene_object_info = get_scene_object_info(
            chunk=self.chunked_data[idx],
            chunk_size=data["chunk_size"],
            actual_chunk_size=data["actual_chunk_size"],
            tokenizer=self.tokenizer,
            max_length=static_cfg.data.token_max_count,
        )
        data.update(scene_object_info)

        return data
