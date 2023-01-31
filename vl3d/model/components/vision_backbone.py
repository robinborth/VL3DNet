"""
The module for the backbone mapper.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn


class VisionBackbone(pl.LightningModule):
    def __init__(
        self,
        freeze: bool = False,
        use_bbox: bool = True,
        dim: int = 768,
        act_fn: torch.nn.Module = None,
        norm: bool = False,
        dropout: float = 0.0,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["act_fn"])

        in_features = 32 + self.hparams.use_bbox * 6
        self.dense = nn.Linear(in_features=in_features, out_features=self.hparams.dim)
        self.act_fn = act_fn
        if self.hparams.norm:
            self.norm = nn.LayerNorm(self.hparams.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Takes the proposals features and maps them to the same as bert embeddings."""
        x = self.dense(x)
        x = self.act_fn(x)
        if self.hparams.norm:
            x = self.norm(x)
        x = self.dropout(x)
        return x

    def configure_optimizers(self):
        if self.hparams.freeze:
            return None
        optimizer = {} if self.hparams.optimizer is None else self.hparams.optimizer
        return {"params": self.parameters(), **optimizer}
