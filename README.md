# Increasing Width Allows Greedy Layer-wise Training to Rival End-to-End Backpropagation in Self-Supervised Learning

Anonymous code for the ICLR 2027 submission **Increasing Width Allows Greedy Layer-Wise Training to Rival End-to-End Backpropagation in Self-Supervised Learning**.

Here, we investigate how increasing network width affects greedy layer-wise self-supervised learning in relatively shallow networks. We show that sufficiently wide networks trained one layer at a time can rival end-to-end backpropagation, using Barlow Twins and SimCLR. The code includes training, kNN evaluation, and representational geometry analysis, along with the control experiments and analyses added during review (freezing controls, μP, linear probes, and compute measurements).

Full instructions to run paper experiments are included below:

## Running / Reproduction Instructions:

Use Python 3.10+ with a compatible PyTorch/torchvision installation.

```bash
python -m pip install -r requirements.txt
python train.py
```

**Edit `OBJECTIVE` and `SETTINGS` at the top of `train.py` before running.** Set `OBJECTIVE = "barlow_twins"` (default) or `OBJECTIVE = "simclr"`. Both use the same training loop. `simclr.py` is an imported module.

In `SETTINGS`, choose the training method (`"e2e"` or `"greedy"`, or one of the freezing controls below), dataset (`"cifar10"` or `"cifar100"`), depth, width, seed, and output directory. Objective-specific hyperparameters are in `BARLOW_DEFAULTS` in `train.py` and `DEFAULTS` in `simclr.py`. They both contain the selected paper values. Entries in `SETTINGS` override defaults. `epochs` is the total budget (i.e. 1,000 epochs means 250 per greedy Conv4 layer or 125 per Conv8 layer).

Data automatically downloads to `data/`. Each run needs a new output directory and saves `configuration.json`, `history.csv` (loss and test kNN accuracy), and `final_encoder.pt`.

## Geometry

Set `geometry` to `"final"` or `"trajectory"` in `train.py` to save SSF/SNF measurements to `geometry.csv`. For a saved checkpoint, edit the paths at the top of `geometry.py`, then run:

```bash
python geometry.py
```

## Experiments added during review

All of these use Barlow Twins on CIFAR-10. Training options are set in `SETTINGS` in `train.py`; the analysis scripts have their settings at the top and are run with `python <script>.py`.

- **Freezing controls** (Section 4.4, Appendix A.6). Set `method` to `"progressive_freeze"` (end-to-end training where layer *l* freezes after epoch *l* × epochs / depth, when greedy training would move on) or `"local"` (every layer has its own projector and loss like greedy training, but all layers train at the same time, with gradients stopped between layers).
- **μP** (Appendix A.5). Set `parametrization = "mup"` (needs `pip install mup`). `python coord_check.py` runs the coordinate check. For the learning-rate check, set `learning_rate` to `0.00031623` or `0.0031623` (10^-3.5 and 10^-2.5).
- **Linear probes** (Appendix A.4). `python linear_probe.py` fits the probe on the final layer of a saved encoder; set `LAYERS = "all"` for every layer, or `DATASET = "cifar100"` for the CIFAR-100 transfer probes.
- **Compute and memory** (Appendix A.7). `python compute.py` measures training FLOPs and peak memory for end-to-end and greedy training at one depth and width.
- **Information retention.** `python retention.py` measures how much of each layer's features can be recovered linearly from the next layer.
- **Shorter greedy budgets.** Use `method = "greedy"` with `epochs` set to `240` or `500`.
- **ResNet-18 pilot.** Set `architecture = "resnet18"` (keep `depth = 4`: greedy training treats its four stages as layers).

## Files

- `train.py`: experiment settings, Barlow Twins definitions, and the single training loop for all four methods.
- `simclr.py`: SimCLR defaults, projector, loss, and combined-view handling.
- `architecture.py`: convolutional encoder, CIFAR ResNet-18, their initialization, and checkpoint loading.
- `parametrization.py`: standard parametrization or μP for the encoder, projector, and optimizer.
- `data.py`: CIFAR datasets, augmentations, and data loaders.
- `evaluation.py`: kNN accuracy.
- `seeding.py`: seeds for runs, models, and data-loader workers.
- `geometry.py`: SSF/SNF equations and checkpoint analysis.
- `linear_probe.py`: linear probes on a saved encoder (Kornblith et al., 2019).
- `retention.py`: how much of each layer's features the next layer keeps.
- `compute.py`: training FLOPs and peak memory.
- `coord_check.py`: μP coordinate check.


For the paper's Barlow Twins CIFAR-10 runs, use both methods, seeds 0–2, and widths 0.25, 0.5, 1, 2, 4, 8, 16 (plus 32 for Conv4). SimCLR/CIFAR-10 and Barlow Twins/CIFAR-100 use Conv4, all eight widths, and seed 1. Geometry throughout the run uses separate Conv4 32× seed-1 runs. The Barlow Twins cosine-schedule check is available through `cosine_decay`. The μP sweep and freezing controls use Conv4, all eight widths, and seed 0, with seeds 1–2 added for the freezing controls at 16×. The ResNet-18 pilot uses seed 0 and widths 0.25–4.
