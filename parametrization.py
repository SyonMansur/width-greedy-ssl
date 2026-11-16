# standard parametrization or maximal update parametrization (mup), chosen in train.py
# mup (yang et al., 2022, tensor programs v) scales the initialization and learning rates with
# width so that every width trains like the base width (1x, where we chose our hyperparameters).
# we use the authors' mup package; at 1x, mup and the standard parametrization are identical.
# mup is only imported for mup runs, so the package isn't needed otherwise

import torch
from torch import nn


def readout_layer(settings):
    # the readout is the layer that maps width-scaled features to a fixed size; here that's the
    # first projector layer, which goes from the encoder's channels to the projector dimension
    if settings["parametrization"] == "mup":
        from mup import MuReadout

        return MuReadout
    return nn.Linear


def set_width_shapes(module, build_at_width, settings):
    # mup finds which weight dimensions grow with width by comparing copies of the module at
    # the base width (1x) and at 2x; the copies only supply shapes, so keep the random state
    if settings["parametrization"] != "mup":
        return
    from mup import set_base_shapes

    with torch.random.fork_rng(devices=[]):
        base = build_at_width(1.0)
        delta = build_at_width(2.0)
    set_base_shapes(module, base, delta=delta)


def make_optimizer(parameters, settings):
    # mup's adam divides the learning rate of every hidden weight (both dimensions grow with
    # width) by its width multiplier; all other parameters keep the base learning rate
    if settings["parametrization"] == "mup":
        from mup import MuAdam

        return MuAdam(
            parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
        )
    return torch.optim.Adam(
        parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
