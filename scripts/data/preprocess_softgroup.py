"""
This module preprocesses the scannet dataset through softgroup and extracts the 
features along with the ground truth bounding boxes into pth files, that can be used
for further modules.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from vl3d.config import cfg
from vl3d.data.collate_functions import collate_fn
from vl3d.data.dataset import SceneDataset
from vl3d.data.utils import get_nms_instances
from vl3d.model.components.softgroup import SoftGroup


class PredictProposals(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.softgroup = SoftGroup.load_from_checkpoint(cfg.softgroup.ckpt_path)

    def forward(self, data_dict):
        return self.softgroup(data_dict)

    def predict_step(self, data_dict, idx):
        output_dict = self.forward(data_dict)
        output_dict = self.softgroup._get_pred_instances(
            data_dict["scene_ids"][0],
            data_dict["locs"].cpu().numpy(),
            output_dict["proposals_idx"].cpu(),
            output_dict["semantic_scores"].size(0),
            output_dict["cls_scores"].cpu(),
            output_dict["iou_scores"].cpu(),
            output_dict["mask_scores"].cpu(),
            output_dict["proposals_features"].cpu(),
        )

        data = {}
        data["scene_id"] = data_dict["scene_ids"][0]
        data["locs"] = data_dict["locs"].cpu().numpy()
        data["rgb"] = data_dict["rgb"].cpu().numpy()

        data["gt_bboxes"] = list(zip(data_dict["instance_id"].cpu().numpy(), data_dict["gt_bbox"].cpu().numpy()))

        pred_bboxes, proposals_features, pred_bboxes_conf = [], [], []
        for p in output_dict:
            pred_bboxes.append(p["pred_bbox"])
            proposals_features.append(p["feature"])
            pred_bboxes_conf.append(p["conf"])
        data["pred_bboxes"] = np.array(pred_bboxes)
        data["proposals_features"] = np.array(proposals_features)
        data["pred_bboxes_conf"] = np.array(pred_bboxes_conf)
        nms_idx = (
            get_nms_instances(
                torch.from_numpy(data["pred_bboxes"]),
                torch.from_numpy(data["pred_bboxes_conf"]),
                cfg.softgroup.nms_threshold,
            )
            .cpu()
            .numpy()
        )
        data["pred_bboxes"] = data["pred_bboxes"][nms_idx]
        data["proposals_features"] = data["proposals_features"][nms_idx]
        data["pred_bboxes_conf"] = data["pred_bboxes_conf"][nms_idx]

        return data


def prune_predictions(predictions):
    data = defaultdict(dict)

    for output in predictions:
        scene_id = output["scene_id"]
        if scene_id in data:
            data[scene_id]["gt_bboxes"].extend(output["gt_bboxes"])
        else:
            data[scene_id]["locs"] = output["locs"]
            data[scene_id]["rgb"] = output["rgb"]
            data[scene_id]["gt_bboxes"] = output["gt_bboxes"]
            data[scene_id]["pred_bboxes"] = output["pred_bboxes"]
            data[scene_id]["proposals_features"] = output["proposals_features"]
            data[scene_id]["pred_bboxes_conf"] = output["pred_bboxes_conf"]

    for scene_id in data:
        instance_ids = []
        gt_bboxes = []
        for instance_id, gt_bbox in data[scene_id]["gt_bboxes"]:
            if instance_id not in instance_ids and instance_id != -1:
                instance_ids.append(instance_id)
                gt_bboxes.append(gt_bbox)
        data[scene_id]["gt_bboxes"] = list(zip(instance_ids, gt_bboxes))

    return dict(data)


def predict_and_save_proposals(split):
    dataset = SceneDataset(split=split, debug=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model = PredictProposals()
    trainer = pl.Trainer(accelerator="gpu", devices=[2])
    predictions = trainer.predict(model, dataloaders=dataloader)
    data = prune_predictions(predictions)
    for scene_id in data:
        path = os.path.join(cfg.softgroup.path, split, f"{scene_id}.pth")
        torch.save(data[scene_id], path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    predict_and_save_proposals("train")
    predict_and_save_proposals("val")
