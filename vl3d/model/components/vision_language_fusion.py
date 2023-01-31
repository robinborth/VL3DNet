import pytorch_lightning as pl
import torch
import torch.nn as nn


class VisionLanguageFusionBlock(pl.LightningModule):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 6,
        dim_mlp: int = 3072,
        act_fn: torch.nn.Module = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["act_fn"])

        # Attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hparams.dim,
            num_heads=self.hparams.num_heads,
            batch_first=True,
        )

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.hparams.dim, out_features=self.hparams.dim_mlp),
            act_fn,
            nn.Linear(in_features=self.hparams.dim_mlp, out_features=self.hparams.dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(self.hparams.dim)
        self.norm2 = nn.LayerNorm(self.hparams.dim)
        self.dropout = nn.Dropout(self.hparams.dropout)

    def forward(self, x, **kwargs):
        # Attention part
        attn_out, _ = self.self_attn(x, x, x, **kwargs)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.mlp(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class VisionLanguageFusion(pl.LightningModule):
    def __init__(
        self,
        freeze: bool = False,
        num_blocks: int = 1,
        dim: int = 768,
        num_heads: int = 6,
        dim_mlp: int = 3072,
        act_fn: torch.nn.Module = None,
        dropout: float = 0.0,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["act_fn"])

        blocks = []
        for _ in range(self.hparams.num_blocks):
            blocks.append(
                VisionLanguageFusionBlock(
                    dim=self.hparams.dim,
                    num_heads=self.hparams.num_heads,
                    dim_mlp=self.hparams.dim_mlp,
                    act_fn=act_fn,
                    dropout=self.hparams.dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, visual_input, language_input, **kwargs):
        x = torch.cat((visual_input, language_input), dim=1)
        for layer in self.blocks:
            x = layer(x, **kwargs)
        return x[:, : visual_input.size(1)], x[:, visual_input.size(1) :]

    def configure_optimizers(self):
        if self.hparams.freeze:
            return None
        optimizer = {} if self.hparams.optimizer is None else self.hparams.optimizer
        return {"params": self.parameters(), **optimizer}
