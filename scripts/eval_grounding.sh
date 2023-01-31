#!bin/bash

python eval.py \
eval.mode=0 \
trainer.devices=[0] \
+logger.wandb.tags=["vg","eval"] \
model=grounding