#!bin/bash

python train.py \
data.batch_size=8 \
trainer.devices=[1] \
+logger.wandb.tags=["dc","train"] \
model=captioning
