"""
The main script where the VL3DNet is trained with.
"""
import os
import re
import shutil
from os import listdir

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None | float:
    pl.seed_everything(seed=cfg.general.seed, workers=True)

    print("==> initializing data ...")
    datamodule = hydra.utils.instantiate(cfg.data)

    print("==> initializing model ...")
    model = hydra.utils.instantiate(cfg.model)

    print("==> initializing callbacks ...")
    callbacks = hydra.utils.instantiate(cfg.callbacks)

    print("==> initializing logger...")
    wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)
    # wandb_logger.watch(model, **OmegaConf.to_container(cfg.logger.watch))

    print("==> initializing trainer ...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    print("==> start training ...")
    trainer.fit(model, datamodule)

    print("==> extract best metric")
    metric = trainer.callback_metrics[cfg.general.optimizer_goal].item()

    if cfg.trainer.enable_checkpointing:
        print("==> start checkpointing")

        # finds the best and latest ckeckpoint for the mode
        best_ckpts, exist_best_ckpt, latest_ckpts = [], False, []
        for ckpt in listdir("outputs"):
            if ckpt.startswith(f"latest:mode={cfg.model.mode}"):
                latest_ckpts.append(ckpt)
            if ckpt.startswith(f"best:mode={cfg.model.mode}"):
                exist_best_ckpt = True
                match = re.search(r"val_loss=(.*)\.(ckpt|yaml)", ckpt)
                best_metric = float(match.group(1))
                if metric < best_metric:
                    best_ckpts.append(ckpt)

        # the path to the current config file & the name of the checkpoint
        current_config_path = f"{cfg.general.run_dir}/.hydra/config.yaml"
        file_name = f"mode={cfg.model.mode}-val_loss={metric:.6f}"

        # save the latest checkpoint allways
        trainer.save_checkpoint(f"outputs/latest:{file_name}.ckpt")
        shutil.copyfile(current_config_path, f"outputs/latest:{file_name}.yaml")
        if latest_ckpts:
            for latest_ckpt in latest_ckpts:
                os.remove(f"outputs/{latest_ckpt}")

        # save the best checkpoint
        if not exist_best_ckpt:
            trainer.save_checkpoint(f"outputs/best:{file_name}.ckpt")
            shutil.copyfile(current_config_path, f"outputs/best:{file_name}.yaml")
        elif best_ckpts:
            trainer.save_checkpoint(f"outputs/best:{file_name}.ckpt")
            shutil.copyfile(current_config_path, f"outputs/best:{file_name}.yaml")
            for best_ckpt in best_ckpts:
                os.remove(f"outputs/{best_ckpt}")

    print("==> close wandb connection")
    wandb.finish()

    return metric


if __name__ == "__main__":
    run()
