"""
This code implements a Visual-Language 3D (VL3D) Network using PyTorch Lightning.
The network is designed for visual grounding, i.e. localizing the objects in an image
based on textual descriptions, and joint image-text generation tasks (captioning).
The VL3D network is composed of multiple components including a language backbone,
vision backbone, vision-language fusion, grounding head, classification head, and
captioning head. The model's components are passed in as parameters during instantiation.
The code also contains logging methods to store scene predictions.
"""


from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import wandb
from vl3d.evaluation.metrics import captioning_metrics, grounding_metrics
from vl3d.loss.captioning_loss import CaptioningLoss
from vl3d.loss.classification_loss import ClassificationLoss
from vl3d.loss.localization_loss import LocalizationLoss
from vl3d.model.components.captioning_head import DenseCaptioningHead
from vl3d.model.components.classification_head import ClassificationHead
from vl3d.model.components.grounding_head import VisualGroundingHead
from vl3d.model.components.language_backbone import LanguageBackbone
from vl3d.model.components.vision_backbone import VisionBackbone
from vl3d.model.components.vision_language_fusion import VisionLanguageFusion
from vl3d.model.utils import (
    encode_teacher_sequences,
    greedy_search,
    vision_language_attention_mask,
    vision_language_key_padding_mask,
)
from vl3d.visualize.wandb import create_wandb_grounding_scene


class VL3DNet(pl.LightningModule):
    def __init__(
        self,
        mode: int = 0,
        batch_size: int = 32,
        freeze_bert: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: object = None,
        language_backbone: LanguageBackbone = None,
        vision_backbone: VisionBackbone = None,
        vision_language_fusion: VisionLanguageFusion = None,
        grounding_head: VisualGroundingHead = None,
        classification_head: ClassificationHead = None,
        captioning_head: DenseCaptioningHead = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "language_backbone",
                "vision_backbone",
                "vision_language_fusion",
                "grounding_head",
                "classification_head",
                "captioning_head",
            ]
        )

        self.language_backbone = language_backbone
        self.vision_backbone = vision_backbone
        self.vision_language_fusion = vision_language_fusion
        self.grounding_head = grounding_head
        self.classification_head = classification_head
        self.captioning_head = captioning_head
 

    @property
    def grounding_mode(self) -> bool:
        return self.hparams.mode == 0 or self.hparams.mode == 2

    @property
    def captioning_mode(self) -> bool:
        return self.hparams.mode == 1 or self.hparams.mode == 2

    def _feed(self, data_dict):
        output_dict = {}

        # backbone feature collection
        proposals_features = self.vision_backbone(data_dict["pred_bboxes_features"])

        # visual grounding & joint
        if self.grounding_mode:
            # select the mean of all proposals
            mean_proposals_features = proposals_features.mean(dim=1)[:, None, :]

            # get the language embeddings
            language_embeddings = self.language_backbone(
                input_ids=data_dict["input_ids"],
                attention_mask=data_dict["language_attention_mask"],
            )["last_hidden_state"]
            language_input = language_embeddings + mean_proposals_features

            # vision language fusion
            mask = vision_language_key_padding_mask(
                vision_mask=data_dict["pred_bboxes_attention_mask"],
                language_mask=data_dict["language_attention_mask"],
            )

            enhanced_proposals_features, enhanced_language_features = self.vision_language_fusion(
                visual_input=proposals_features,
                language_input=language_input,
                key_padding_mask=mask,
            )

            # grounding head predictions
            pred_bboxes_logits = self.grounding_head(enhanced_proposals_features)
            output_dict["pred_bboxes_logits"] = pred_bboxes_logits

            # description classification predictions
            cls_token = enhanced_language_features[:, 0, :]
            classification_logits = self.classification_head(cls_token)
            output_dict["classification_logits"] = classification_logits

        # dense captioning & joint
        if self.captioning_mode:
            # select the queried proposal
            pred_idx = torch.argmax(data_dict["pred_bboxes_labels"], dim=1)
            queried_proposal_features = proposals_features[torch.arange(len(pred_idx)), pred_idx][:, None, :]

            # get the language embeddings
            assert data_dict["language_attention_mask"].shape == data_dict["input_ids"].shape
            attention_mask = encode_teacher_sequences(data_dict["language_attention_mask"])
            language_embeddings = self.language_backbone(
                input_ids=data_dict["input_ids"],
                attention_mask=attention_mask,
            )["last_hidden_state"]
            language_input = language_embeddings + queried_proposal_features

            # vision language fusion
            mask = vision_language_attention_mask(
                vision_masks=data_dict["pred_bboxes_attention_mask"],
                language_masks=attention_mask,
            ).repeat_interleave(self.vision_language_fusion.hparams.num_heads, dim=0)

            _, enhanced_language_features = self.vision_language_fusion(
                visual_input=proposals_features,
                language_input=language_input,
                attn_mask=mask,
            )

            # grounding head predictions
            captioning_logits = self.captioning_head(enhanced_language_features)
            output_dict["captioning_logits"] = captioning_logits

        return output_dict

    def forward(self, data_dict):
        output_dict = self._feed(data_dict)

        if self.grounding_mode:
            idx = output_dict["pred_bboxes_logits"].argmax(dim=1)
            running_idx = torch.arange(data_dict["pred_bboxes"].shape[0])
            output_dict["pred_bbox"] = data_dict["pred_bboxes"][running_idx, idx]
            output_dict["pred_bbox_score"] = F.softmax(output_dict["pred_bboxes_logits"], dim=1)[running_idx, idx]
            output_dict["pred_sem_label"] = output_dict["classification_logits"].argmax(dim=1)

        if self.captioning_mode:
            output_dict["pred_caption_ids"] = greedy_search(forward=self._feed, data_dict=data_dict)
            output_dict["pred_caption"] = self.language_backbone.batch_decode(output_dict["pred_caption_ids"])
            output_dict["gt_caption"] = self.language_backbone.batch_decode(data_dict["input_ids"])

        return output_dict

    def _loss(self, data_dict, output_dict):
        losses = {}
        total_loss = 0

        if self.grounding_mode:
            """localization_loss"""
            localization_criterion = LocalizationLoss()
            localization_loss = localization_criterion(
                output_dict["pred_bboxes_logits"],
                data_dict["pred_bboxes_labels"],
            )
            losses["localization_loss"] = localization_loss
            total_loss += losses["localization_loss"]

            """classification_loss"""
            classification_criterion = ClassificationLoss()
            classification_loss = classification_criterion(
                output_dict["classification_logits"],
                data_dict["gt_sem_label"],
            )
            losses["classification_loss"] = classification_loss
            total_loss += losses["classification_loss"]

        if self.captioning_mode:
            actual_token_length = data_dict["language_attention_mask"].sum(dim=1)
            captioning_criterion = CaptioningLoss()
            captioning_loss = captioning_criterion(
                output_dict["captioning_logits"],
                data_dict["input_ids"],
                actual_token_length,
            )
            losses["captioning_loss"] = captioning_loss
            total_loss += losses["captioning_loss"]

        return total_loss, losses

    def configure_optimizers(self):
        params = [
            self.language_backbone.configure_optimizers(),
            self.vision_backbone.configure_optimizers(),
            self.vision_language_fusion.configure_optimizers(),
            self.grounding_head.configure_optimizers(),
            self.classification_head.configure_optimizers(),
        ]
        optimizer = self.hparams.optimizer(params=[param for param in params if param is not None])

        if self.hparams.lr_scheduler is None:
            return optimizer

        self.hparams.lr_scheduler.scheduler = self.hparams.lr_scheduler.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": dict(self.hparams.lr_scheduler),
        }

    def training_step(self, data_dict, batch_idx):
        output_dict = self._feed(data_dict)
        total_loss, losses = self._loss(data_dict, output_dict)

        self.log("train/total_loss", total_loss, prog_bar=True, sync_dist=True)
        for key, specific_loss in losses.items():
            self.log(f"train/{key}", specific_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)

        return total_loss

    def validation_step(self, data_dict, idx):
        output_dict = self._feed(data_dict)
        total_loss, losses = self._loss(data_dict, output_dict)

        self.log("val/total_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        for key, specific_loss in losses.items():
            self.log(f"val/{key}", specific_loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        output_dict.update(data_dict)

        output_dict.update(data_dict)
        if self.grounding_mode:
            for name, score in grounding_metrics(output_dict):
                self.log(f"test/{name}", score, sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)
        if self.captioning_mode:
            for name, score in captioning_metrics(output_dict):
                self.log(f"test/{name}", score, sync_dist=True, batch_size=self.hparams.batch_size, prog_bar=True)

            # count the predicted bboxes and gt bboxes during testing for the final metrics
            total_count = output_dict["pred_bboxes_attention_mask"].size(1)
            pred_bbox_count = total_count - output_dict["pred_bboxes_attention_mask"][idx].sum()
            self.log("test/pred_count", pred_bbox_count, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/gt_count", float(len(data_dict["scene_id"])), reduce_fx=torch.sum, sync_dist=True)

        # log the first scene results in wandb 
        if idx == 0:
            return output_dict

    def test_epoch_end(self, outputs):
        if not outputs:
            return

        output_dict = outputs[0]

        if self.grounding_mode:
            point_cloud = create_wandb_grounding_scene(output_dict=output_dict, idx=0)
            self.logger.experiment.log({"point_cloud": point_cloud})

        if self.captioning_mode:
            data = list(zip(output_dict["pred_caption"], output_dict["gt_caption"]))
            caption_table = wandb.Table(columns=["prediction", "reference"], data=data)
            self.logger.experiment.log({"caption_table": caption_table})
