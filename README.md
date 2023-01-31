# Joint Architecture for 3D Vision Language

[embed]https://github.com/robinborth/VL3DNet/blob/main/docs/benchmark.pdf[/embed]

A full transformer-based joint architecture for the visual grounding and dense captioning task.

## Introduction

We propose a novel speaker-listener architecture based on D3Net for the dense captioning (DC) and visual grounding (VG) tasks where all submodules are transformer-based. The performance of state-of-the-art speaker-listener architecture relies heavily on the shared detector, which performs poorly in distinguishing nearby instances. In addition, the only shared input of two downstream tasks is raw proposals from the detector, while the relations among them are exploited separately for captioning and grounding heads. To tackle these two problems, our contribution is two-fold: (1) Improve the initial proposals by integrating Mask3D into the detection sub-module. (2) Introduce a shared feature enhancement module before the task-specific speaker and listener, which allows one to learn the relations between proposals uniformly.

## Requirements

- python>=3.10
- cuda>=11.6


## Scripts

There are different options to run the project. Because we use hydra we can changee the configurations
for the different training/prediction modes via cli.

### Training

For training there are several options you can choose, depending on the task you want to train on or 
which task you want to finetune.

To train the visual grounding task just use:
```bash
python train.py model=grounding
```

To train the dense captioning task just use:
```bash
python train.py model=captioning
```

To train both tasks jointly use:
```bash
python train.py model=vl3dnet
```

### Evaluating

For evaluating there are several options you can choose.

To evaluate the visual grounding task just use:

```bash
. scripts/eval_grounding.sh
```

To evaluate the dense captioning task just use:
```bash
. scripts/eval_captioning.sh
```

To evaluate both tasks jointly use:
```bash
. scripts/eval_vl3dnet.sh
```

### Hyparams Search

```bash
python train.py +hparams_search=optuna
```

## License

Copyright (c) 2022 Yaomengxi Han, Robin Borth
