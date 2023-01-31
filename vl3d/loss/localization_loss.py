"""
The localization loss for the grounding task.
"""


import pytorch_lightning as pl
import torch.nn as nn


class LocalizationLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_bboxes_confidences, pred_bboxes_labels):
        """The localization loss for the grounding head."""
        return self.criterion(pred_bboxes_confidences, pred_bboxes_labels)
