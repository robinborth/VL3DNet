import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from minsu3d.common_ops.functions import common_ops, softgroup_ops
from minsu3d.evaluation.instance_segmentation import rle_encode
from minsu3d.loss import (
    ClassificationLoss,
    IouScoringLoss,
    MaskScoringLoss,
    PTOffsetLoss,
    SemSegLoss,
)
from minsu3d.model.general_model import clusters_voxelization, get_batch_offsets
from minsu3d.model.module import Backbone, TinyUnet


class SoftGroup(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()

        self.instance_classes = data.classes - len(data.ignore_classes)

        input_channel = model.use_coord * 3 + model.use_color * 3 + model.use_normal * 3 + model.use_multiview * 128
        self.backbone = Backbone(
            input_channel=input_channel,
            output_channel=model.m,
            block_channels=model.blocks,
            block_reps=model.block_reps,
            sem_classes=data.classes,
        )

        output_channel = model.m
        self.tiny_unet = TinyUnet(output_channel)
        self.classification_branch = nn.Linear(
            output_channel,
            self.instance_classes + 1,
        )
        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, self.instance_classes + 1),
        )
        self.iou_score = nn.Linear(output_channel, self.instance_classes + 1)

    def forward(self, data_dict):
        data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"], data_dict["p2v_map"])  # (M, C)
        output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])

        semantic_scores = output_dict["semantic_scores"].softmax(dim=-1)
        batch_idxs = data_dict["vert_batch_ids"]

        proposals_offset_list = []
        proposals_idx_list = []

        for class_id in range(self.hparams.data.classes):
            if class_id in self.hparams.data.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.hparams.model.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.hparams.model.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs].int()
            batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
            coords_ = data_dict["locs"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]
            idx, start_len = common_ops.ballquery_batch_p(
                coords_ + pt_offsets_,
                batch_idxs_,
                batch_offsets_,
                self.hparams.model.grouping_cfg.radius,
                self.hparams.model.grouping_cfg.mean_active,
            )

            proposals_idx, proposals_offset = softgroup_ops.sg_bfs_cluster(
                self.hparams.data.point_num_avg,
                idx.cpu(),
                start_len.cpu(),
                self.hparams.model.grouping_cfg.npoint_thr,
                class_id,
            )
            proposals_idx[:, 1] = object_idxs.cpu()[proposals_idx[:, 1].long()].int()

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)

        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)

        if proposals_offset.shape[0] > self.hparams.model.train_cfg.max_proposal_num:
            proposals_offset = proposals_offset[: self.hparams.model.train_cfg.max_proposal_num + 1]
            proposals_idx = proposals_idx[: proposals_offset[-1]]
            assert proposals_idx.shape[0] == proposals_offset[-1]

        proposals_offset = proposals_offset.cuda()
        output_dict["proposals_idx"] = proposals_idx
        output_dict["proposals_offset"] = proposals_offset

        inst_feats, inst_map = clusters_voxelization(
            clusters_idx=proposals_idx,
            clusters_offset=proposals_offset,
            feats=output_dict["point_features"],
            coords=data_dict["locs"],
            mode=4,
            device=self.device,
            **self.hparams.model.instance_voxel_cfg,
        )
        inst_map = inst_map.long().cuda()

        feats = self.tiny_unet(inst_feats)

        # predict mask scores
        mask_scores = self.mask_scoring_branch(feats.features)
        output_dict["mask_scores"] = mask_scores[inst_map]
        output_dict["instance_batch_idxs"] = feats.coordinates[:, 0][inst_map]

        # predict instance cls and iou scores
        global_feats = self.global_pool(feats)
        output_dict["cls_scores"] = self.classification_branch(global_feats)
        output_dict["iou_scores"] = self.iou_score(global_feats)

        # getting the features for the proposals
        output_dict["proposals_features"] = global_feats
        # getting the pred_bboxes from the proposals
        output_dict["pred_bboxes"] = self._get_pred_bboxes(
            locs=data_dict["locs"],
            proposals_idx=proposals_idx,
            num_points=semantic_scores.size(0),
            cls_scores=output_dict["cls_scores"],
            mask_scores=output_dict["mask_scores"],
        )

        return output_dict

    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def _loss(self, data_dict, output_dict):
        losses = {}
        total_loss = 0

        proposals_idx = output_dict["proposals_idx"][:, 1].cuda()
        proposals_offset = output_dict["proposals_offset"]

        # calculate iou of clustered instance
        ious_on_cluster = common_ops.get_mask_iou_on_cluster(
            proposals_idx,
            proposals_offset,
            data_dict["instance_ids"],
            data_dict["instance_num_point"],
        )

        # filter out background instances
        fg_inds = data_dict["instance_sem_label"] != self.hparams.data.ignore_label
        fg_instance_sem_label = data_dict["instance_sem_label"][fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals,), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.hparams.model.train_cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.hparams.data.ignore_label)
        semantic_loss = sem_seg_criterion(output_dict["semantic_scores"], data_dict["sem_labels"].long())
        losses["semantic_loss"] = semantic_loss

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.hparams.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(
            output_dict["point_offsets"], gt_offsets, valid_mask=valid
        )
        losses["offset_norm_loss"] = offset_norm_loss
        losses["offset_dir_loss"] = offset_dir_loss

        """classification loss"""
        # follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_sem_label.new_full((num_proposals,), self.instance_classes)
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_sem_label[assigned_gt_inds[pos_inds]]
        classification_criterion = ClassificationLoss()
        labels = labels.long()
        classification_loss = classification_criterion(output_dict["cls_scores"], labels)
        losses["classification_loss"] = classification_loss

        """mask scoring loss"""
        mask_cls_label = labels[output_dict["instance_batch_idxs"].long()]
        slice_inds = torch.arange(
            0,
            mask_cls_label.size(0),
            dtype=torch.long,
            device=mask_cls_label.device,
        )
        mask_scores_sigmoid_slice = output_dict["mask_scores"].sigmoid()[slice_inds, mask_cls_label]
        mask_label, mask_label_mask = common_ops.get_mask_label(
            proposals_idx,
            proposals_offset,
            data_dict["instance_ids"],
            data_dict["instance_sem_label"],
            data_dict["instance_num_point"],
            ious_on_cluster,
            self.hparams.data.ignore_label,
            self.hparams.model.train_cfg.pos_iou_thr,
        )

        mask_scoring_criterion = MaskScoringLoss(weight=mask_label_mask, reduction="sum")
        mask_scoring_loss = mask_scoring_criterion(mask_scores_sigmoid_slice, mask_label.float())
        mask_scoring_loss /= torch.count_nonzero(mask_label_mask) + 1
        losses["mask_scoring_loss"] = mask_scoring_loss

        """iou scoring loss"""
        ious = common_ops.get_mask_iou_on_pred(
            proposals_idx,
            proposals_offset,
            data_dict["instance_ids"],
            data_dict["instance_num_point"],
            mask_scores_sigmoid_slice.detach(),
        )
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = labels < self.instance_classes
        iou_score_slice = output_dict["iou_scores"][slice_inds, labels]
        iou_scoring_criterion = IouScoringLoss(reduction="none")
        iou_scoring_loss = iou_scoring_criterion(iou_score_slice, gt_ious)
        iou_scoring_loss = iou_scoring_loss[iou_score_weight].sum() / (iou_score_weight.count_nonzero() + 1)
        losses["iou_scoring_loss"] = iou_scoring_loss

        total_loss += (
            self.hparams.model.loss_weight[0] * semantic_loss
            + self.hparams.model.loss_weight[1] * offset_norm_loss
            + self.hparams.model.loss_weight[3] * classification_loss
            + self.hparams.model.loss_weight[4] * mask_scoring_loss
            + self.hparams.model.loss_weight[5] * iou_scoring_loss
        )

        return losses, total_loss

    def _get_pred_instances(
        self,
        scene_id,
        gt_xyz,
        proposals_idx,
        num_points,
        cls_scores,
        iou_scores,
        mask_scores,
        proposals_features,
    ):
        num_instances = cls_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        cls_pred_list, score_pred_list, mask_pred_list, features_pred_list = [], [], [], []
        for i in range(self.instance_classes):
            cls_pred = cls_scores.new_full((num_instances,), i, dtype=torch.long)
            cur_cls_scores = cls_scores[:, i]
            cur_iou_scores = iou_scores[:, i]
            cur_mask_scores = mask_scores[:, i]
            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.bool, device="cpu")
            mask_inds = cur_mask_scores > self.hparams.model.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds].long()
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = True

            # filter low score instance
            inds = cur_cls_scores > self.hparams.model.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]
            features_pred = proposals_features[torch.Tensor(inds)]

            # filter too small instances
            npoint = torch.count_nonzero(mask_pred, dim=1)
            inds = npoint >= self.hparams.model.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]
            features_pred = features_pred[torch.Tensor(inds)]

            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
            features_pred_list.append(features_pred)

        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()
        features_pred = torch.cat(features_pred_list).numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {
                "scene_id": scene_id,
                "label_id": cls_pred[i],
                "conf": score_pred[i],
                "pred_mask": rle_encode(mask_pred[i]),
                "feature": features_pred[i],
            }
            pred_xyz = gt_xyz[mask_pred[i]]
            pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
        return pred_instances

    def _get_pred_bboxes(
        self,
        locs,
        proposals_idx,
        num_points,
        cls_scores,
        mask_scores,
    ):
        num_instances = cls_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        proposals_idx = proposals_idx.to(device=locs.device)

        for i in range(self.instance_classes):
            cur_mask_scores = mask_scores[:, i]
            mask_pred = locs.new_full((num_instances, num_points), fill_value=0, dtype=torch.bool)
            mask_inds = cur_mask_scores > self.hparams.model.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds].long()
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = True

        pred_bboxes = []
        for i in range(len(mask_pred)):
            pred_xyz = locs[mask_pred[i]]
            if not len(pred_xyz):
                pred_bbox = pred_xyz.new_full((6,), fill_value=0)
            else:
                pred_bbox = torch.concat((pred_xyz.min(0).values, pred_xyz.max(0).values))
            pred_bboxes.append(pred_bbox)

        pred_bboxes = torch.stack(pred_bboxes, dim=0)
        return pred_bboxes

    def predict_step(self, data_dict):
        output_dict = self(data_dict)
        pred_instances = self._get_pred_instances(output_dict)
        return pred_instances
