import pytorch_lightning as pl
import torch
import torch.nn as nn


class ObjectDetctionLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cross_entropy_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss(reduction="mean")

    def forward(self, location_confidence, pred_bboxes, gt_bbox, sem_scores, sem_label, num_classes):
        """The localization loss for the softgroup backbone."""

        sem_score, pred_bbox = (
            sem_scores[torch.argmax(location_confidence)],
            pred_bboxes[torch.argmax(location_confidence)],
        )

        sem_target = torch.zeros(num_classes)
        sem_target[sem_label] = 1
        sem_score_wo_bg = sem_score[:-1]
        sem_target = sem_target.to(device=sem_score_wo_bg.device)
        sem_cls_loss = self.cross_entropy_criterion(sem_score_wo_bg, sem_target)

        box_reg_loss = self.regression_criterion(pred_bbox, gt_bbox)

        localization_loss = sem_cls_loss * 0.1 + box_reg_loss

        return localization_loss
