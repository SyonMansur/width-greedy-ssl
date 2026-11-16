# training compute and memory of end-to-end and greedy training for one network
# flops are counted by pytorch over the forward and backward pass of one training step, for
# every stage (convolutions and matrix multiplies at 2 flops per multiply-add; batch norm, relu,
# and adam's updates aren't counted, as is standard). memory is the peak gpu memory allocated
# during a training step, without evaluation. no training happens, so this runs in minutes

import torch
from torch.utils.flop_counter import FlopCounterMode

from architecture import make_encoder
from train import BARLOW_DEFAULTS, SETTINGS, barlow_batch_loss, make_barlow_projector, start_stage


# set the network here, then run python compute.py (needs a gpu)
DEPTH = 4
WIDTH = 32.0
EPOCHS = 1000
STEPS_PER_EPOCH = 50000 // SETTINGS["batch_size"]  # training drops the last incomplete batch


def measure_step(settings, layer, device):
    # set up one training stage exactly as train.py does
    encoder = make_encoder(settings, device)
    projector, optimizer = start_stage(encoder, layer, settings, make_barlow_projector, device)
    greedy = settings["method"] == "greedy"
    # flops and memory don't depend on pixel values, so random images will do
    first, second = torch.randn(2, settings["batch_size"], 3, 32, 32, device=device)

    def loss():
        return barlow_batch_loss(encoder, projector, first, second, layer, greedy, settings)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()

    # two warm-up steps create adam's state and let cudnn pick its kernels
    step()
    step()
    # then record the peak memory of an ordinary step
    torch.cuda.reset_peak_memory_stats()
    step()
    memory = torch.cuda.max_memory_allocated()
    # count the flops of one more forward and backward pass
    optimizer.zero_grad(set_to_none=True)
    counter = FlopCounterMode(display=False)
    with counter:
        loss().backward()
    return counter.get_total_flops(), memory


if __name__ == "__main__":
    # use the same gpu settings as training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    totals = {}
    for method in ("e2e", "greedy"):
        settings = {**SETTINGS, **BARLOW_DEFAULTS, "method": method, "depth": DEPTH, "width": WIDTH}
        # greedy training has one stage per layer, each with an equal share of the epochs
        stages = range(1, DEPTH + 1) if method == "greedy" else [DEPTH]
        epochs_per_stage = EPOCHS // len(stages)
        totals[method] = 0
        for layer in stages:
            flops, memory = measure_step(settings, layer, device)
            totals[method] += flops * STEPS_PER_EPOCH * epochs_per_stage
            print(
                f"{method}, layer {layer}: {flops / 1e12:.2f} TFLOP per step, "
                f"{memory / 2**30:.2f} GiB peak memory",
                flush=True,
            )
        print(f"{method}: {totals[method] / 1e15:,.0f} PFLOP over {EPOCHS} epochs", flush=True)
    # the end-to-end epoch at which training has used as much compute as the whole greedy run
    e2e_per_epoch = totals["e2e"] / EPOCHS
    print(f"End-to-end training uses {totals['e2e'] / totals['greedy']:.1f}x the flops of greedy")
    print(
        f"End-to-end matches the greedy run's flops at epoch {totals['greedy'] / e2e_per_epoch:.0f}"
    )
