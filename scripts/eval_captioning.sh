#!bin/bash

python eval.py \
eval.mode=1 \
trainer.devices=[2] \
+logger.wandb.tags=["dc","eval"] \
model=captioning
