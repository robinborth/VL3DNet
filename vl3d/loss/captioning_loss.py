"""
The captioning loss for the dense captioning task.
"""


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, sequence_logits, gt_token_ids, actual_token_lengths):
        """The captioning loss for the captioning head.

        Args:
            sequence_confidence: The raw token confidence which is the output of the captiongin head.
            gt_token_id: The input_id for the masked token which we want to predict.

        Returns:
            The cross entropy loss for the logits of the captioning head.
        """
        batch_size, _, vocab_size = sequence_logits.shape
        captioning_loss = 0
        sequence_logits = sequence_logits[:,:-1,]
        targets = F.one_hot(gt_token_ids.to(int), num_classes=vocab_size).to(torch.float32)[:,1:,]
        # captioning_loss = self.criterion(sequence_logits, target)
        for idx in range(batch_size):
            actual_length = actual_token_lengths[idx]
            logits = sequence_logits[idx][:actual_length]
            target = targets[idx][:actual_length]
            batch_loss = self.criterion(logits, target)
            captioning_loss += batch_loss
        captioning_loss /= batch_size
        return captioning_loss
