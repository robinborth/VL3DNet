"""
The module for the backbone mapper.
"""
import pytorch_lightning as pl
import torch
from transformers import BertModel, BertTokenizer, logging


class LanguageBackbone(pl.LightningModule):
    def __init__(
        self,
        freeze: bool = True,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        logging.set_verbosity_error()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if self.hparams.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """Takes the proposals features and maps them to the same as bert embeddings."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def configure_optimizers(self):
        if self.hparams.freeze:
            return None

        optimizer = {} if self.hparams.optimizer is None else self.hparams.optimizer
        return {"params": self.parameters(), **optimizer}

    def batch_decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
