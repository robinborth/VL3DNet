"""
The localization loss for the grounding task.
"""


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, classification_confidences, sem_labels, num_classes: int = 18):
        """The localization loss for the grounding head."""
        one_hot_labels = F.one_hot(sem_labels.to(int), num_classes=num_classes).to(torch.float32)
        return self.criterion(classification_confidences, one_hot_labels)
