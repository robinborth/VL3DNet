#!bin/bash

python train.py \
data.batch_size=8 \
trainer.devices=[2] \
+logger.wandb.tags=["vl3d","train"] \
model=vl3dnet
