# coordinate check for the mup setup (yang et al., 2022, tensor programs v, figure 5)
# end-to-end conv4 networks of each width take a few adam steps on the same batches, and we
# measure how far each layer's convolution output on a fixed set of test images moves from where
# it started, std(x_t - x_0). under mup this change should stay about the same at every width;
# under the standard parametrization it grows with width in the hidden layers (2 to 4)

import numpy as np
import torch

from architecture import make_encoder
from data import make_loaders
from train import BARLOW_DEFAULTS, SETTINGS, barlow_batch_loss, make_barlow_projector, start_stage


# set these, then run python coord_check.py (needs a gpu)
WIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SEEDS = (0, 1, 2)
STEPS = 4  # adam steps, as in tensor programs v
CHECK_IMAGES = 128  # fixed test images whose layer outputs we follow


def conv_outputs(encoder, images):
    # each layer's convolution output, before batch norm, for the fixed images
    outputs = []
    maps = images
    with torch.no_grad():
        for layer in encoder.layers:
            outputs.append(layer[0](maps))
            maps = layer(maps)
    return outputs


def output_change(parametrization, width, seed, batches, images, device):
    # train one network for a few steps and return how much each layer's output moved
    settings = {**SETTINGS, **BARLOW_DEFAULTS, "method": "e2e", "depth": 4, "width": width}
    settings.update(seed=seed, parametrization=parametrization)
    encoder = make_encoder(settings, device)
    projector, optimizer = start_stage(encoder, 4, settings, make_barlow_projector, device)
    start = conv_outputs(encoder, images)
    for first, second in batches:
        optimizer.zero_grad(set_to_none=True)
        barlow_batch_loss(encoder, projector, first, second, 4, False, settings).backward()
        optimizer.step()
    end = conv_outputs(encoder, images)
    return [(after - before).std().item() for before, after in zip(start, end)]


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    # every network trains on the same first few batches and is checked on the same images
    train_loader, _, test_loader = make_loaders({**SETTINGS, "seed": 0}, "barlow_twins")
    batches = []
    for (first, second), _labels in train_loader:
        batches.append((first.to(device), second.to(device)))
        if len(batches) == STEPS:
            break
    images = next(iter(test_loader))[0][:CHECK_IMAGES].to(device)
    for parametrization in ("standard", "mup"):
        # average the change in each layer over seeds
        change = {}
        for width in WIDTHS:
            runs = [
                output_change(parametrization, width, s, batches, images, device) for s in SEEDS
            ]
            change[width] = np.mean(runs, axis=0)
            print(f"{parametrization}, width {width:g}: {np.round(change[width], 4)}", flush=True)
        # under mup, the widest network's change should stay close to the base width's
        growth = change[max(WIDTHS)] / change[1.0]
        print(
            f"{parametrization}: change at {max(WIDTHS):g}x / change at 1x = {np.round(growth, 2)}"
        )
