"""
The datamodlue to load the scene dataset.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from vl3d.data.collate_functions import softgroup_collate_fn
from vl3d.data.dataset import SoftgroupSceneDataset


class SceneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        token_max_count: int = 60,
        proposal_max_count: int = 60,
        drop_sample_iou_threshold: float = 0.0,
        use_bbox: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        val_hparams = self.hparams.copy()
        val_hparams["drop_sample_iou_threshold"] = 0.0
        self.val_set = SoftgroupSceneDataset(
            split="val",
            token_max_count=self.hparams.token_max_count,
            proposal_max_count=self.hparams.proposal_max_count,
            use_bbox=self.hparams.use_bbox,
            drop_sample_iou_threshold=self.hparams.drop_sample_iou_threshold,
        )
        if stage == "fit" or stage is None:
            self.train_set = SoftgroupSceneDataset(
                split="train",
                token_max_count=self.hparams.token_max_count,
                proposal_max_count=self.hparams.proposal_max_count,
                use_bbox=self.hparams.use_bbox,
                drop_sample_iou_threshold=self.hparams.drop_sample_iou_threshold,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=softgroup_collate_fn,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=softgroup_collate_fn,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
