"""
The main script where the VL3DNet is trained with.
"""
from os import listdir

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from vl3d.evaluation.metrics import f1_captioning_metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None | float:
    pl.seed_everything(seed=cfg.general.seed, workers=True)

    print("==> load the checkpoint...")
    checkpoint, cfg_ckpt = None, None
    for ckpt in listdir("outputs"):
        if ckpt.startswith(f"{cfg.eval.ckpt}:mode={cfg.eval.mode}") and ckpt.endswith(".ckpt"):
            checkpoint = torch.load(f"outputs/{ckpt}")
        if ckpt.startswith(f"{cfg.eval.ckpt}:mode={cfg.eval.mode}") and ckpt.endswith(".yaml"):
            cfg_ckpt = OmegaConf.load(f"outputs/{ckpt}")
    if checkpoint is None or cfg_ckpt is None:
        raise Exception(f"No checkpoint for the current mode={cfg.model.mode}")

    print("==> initializing data ...")
    datamodule = hydra.utils.instantiate(cfg.data)

    print("==> initializing model ...")
    model = hydra.utils.instantiate(cfg_ckpt.model, batch_size=8)
    model.load_state_dict(checkpoint["state_dict"])

    print("==> initializing logger...")
    wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)

    print("==> initializing trainer ...")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger)

    print("==> start training ...")
    m = trainer.test(model, datamodule)[0]

    metrics = {}
    if cfg.eval.mode == 0 or cfg.eval.mode == 2:
        metrics["Unique@0.25IoU"] = m["test/acc@0.25IoU_unique"]
        metrics["Unique@0.5IoU"] = m["test/acc@0.5IoU_unique"]
        metrics["Multiple@0.25IoU"] = m["test/acc@0.25IoU_multiple"]
        metrics["Multiple@0.5IoU"] = m["test/acc@0.5IoU_multiple"]
        metrics["Overall@0.25IoU"] = m["test/acc@0.25IoU_overall"]
        metrics["Overall@0.5IoU"] = m["test/acc@0.5IoU_overall"]
    if cfg.eval.mode == 1 or cfg.eval.mode == 2:
        metrics["B@0.5IoU"] = f1_captioning_metrics(m["test/B@0.5IoU"], m["test/gt_count"], m["test/pred_count"])
        metrics["R@0.5IoU"] = f1_captioning_metrics(m["test/R@0.5IoU"], m["test/gt_count"], m["test/pred_count"])
        metrics["M@0.5IoU"] = f1_captioning_metrics(m["test/M@0.5IoU"], m["test/gt_count"], m["test/pred_count"])
        metrics["C@0.5IoU"] = f1_captioning_metrics(m["test/C@0.5IoU"], m["test/gt_count"], m["test/pred_count"])
    df = pd.DataFrame(index=metrics.keys(), data=metrics.values(), columns=["score"])
    df.to_csv(f"results/mode_{cfg.model.mode}.csv")

    print("\n\n==> print metrics results ...")
    print(df)
    print("\n\n")


if __name__ == "__main__":
    run()
