"""
The module for the visual grounding head.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn


class VisualGroundingHead(pl.LightningModule):
    def __init__(
        self,
        freeze: bool = False,
        dim: int = 768,
        act_fn: torch.nn.Module = None,
        norm: bool = False,
        dropout: float = 0.0,
        dim_mlp: int = 768,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["act_fn"])

        self.dense = nn.Linear(in_features=dim, out_features=dim_mlp)
        self.act_fn = act_fn
        if self.hparams.norm:
            self.norm = nn.LayerNorm(self.hparams.dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(in_features=dim_mlp, out_features=1)

    def forward(self, x):
        """Takes the proposals and predicts the proposal_mask.

        Args:
            visual_input: The enrichted proposals with shape (proposal_max_length, dim)
        """
        x = self.dense(x)
        x = self.act_fn(x)
        if self.hparams.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x.view(x.size(0), -1)

    def configure_optimizers(self):
        if self.hparams.freeze:
            return None
        optimizer = {} if self.hparams.optimizer is None else self.hparams.optimizer
        return {"params": self.parameters(), **optimizer}
