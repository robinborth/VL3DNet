"""
The module for dense captioning head.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn


class DenseCaptioningHead(pl.LightningModule):
    def __init__(
        self,
        freeze: bool = False,
        dim: int = 768,
        act_fn: torch.nn.Module = None,
        norm: bool = False,
        dropout: float = 0.0,
        vocalbulary_size: int = 30552,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["act_fn"])

        self.dense = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.act_fn = act_fn
        if self.hparams.norm:
            self.norm = nn.LayerNorm(self.hparams.dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(in_features=dim, out_features=vocalbulary_size)

    def forward(self, x):
        """Creates logits for each vocabualry in bert.

        Args:
            enhanced_language_embedding: One embedding from the language_embeddings of dim (1, dim)
        """
        x = self.dense(x)
        x = self.act_fn(x)
        if self.hparams.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        if self.hparams.freeze:
            return None
        optimizer = {} if self.hparams.optimizer is None else self.hparams.optimizer
        return {"params": self.parameters(), **optimizer}
