#!bin/bash

python train.py \
data.batch_size=8 \
trainer.devices=[0] \
+logger.wandb.tags=["vg","train"] \
model=grounding
