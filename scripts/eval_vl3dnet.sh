#!bin/bash

python eval.py \
eval.mode=2 \
trainer.devices=[2] \
+logger.wandb.tags=["vl3d","eval"] \
model=vl3dnet
