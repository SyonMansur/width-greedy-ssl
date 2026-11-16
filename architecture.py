# encoders used by both barlow twins and simclr

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

from parametrization import set_width_shapes
from seeding import model_seed


# base channel counts and downsampling steps for the two network depths
CHANNELS = {4: (32, 64, 128, 256), 8: (32, 32, 64, 64, 128, 128, 256, 256)}
STRIDES = {4: (1, 2, 2, 2), 8: (1, 1, 2, 1, 2, 1, 2, 1)}

# resnet-18's four stages, each trained as one greedy layer
RESNET18_CHANNELS = (64, 128, 256, 512)
RESNET18_STRIDES = (1, 2, 2, 2)


class LayerwiseEncoder(nn.Module):
    # a list of layers that greedy training can visit one at a time, each followed by pooling

    def features(self, images, stop_layer=None, greedy=False):
        # stop at the active layer during greedy training, or use the full network
        stop_layer = self.depth if stop_layer is None else stop_layer
        maps = images
        if greedy:
            # pass through the frozen layers without tracking their gradients
            with torch.no_grad():
                for layer in self.layers[: stop_layer - 1]:
                    maps = layer(maps)
            # track gradients for the layer we're training now
            maps = self.layers[stop_layer - 1](maps)
        else:
            for layer in self.layers[:stop_layer]:
                maps = layer(maps)
        # average each feature map into one value per channel
        return maps.mean(dim=(2, 3))

    def layer_features(self, images, local=False):
        # pool the output of every layer, not just the last one
        # with local=True, each layer gets a detached input, so it only learns from the loss
        # on its own output (simultaneous local training)
        pooled = []
        maps = images
        for layer in self.layers:
            maps = layer(maps.detach() if local else maps)
            pooled.append(maps.mean(dim=(2, 3)))
        return pooled


class Encoder(LayerwiseEncoder):
    # stack convolution, batch norm, and relu, then pool the spatial dimensions

    def __init__(self, depth, width):
        super().__init__()
        self.depth = depth
        # widen every layer by the same multiplier
        self.channels = [round(c * width) for c in CHANNELS[depth]]
        self.layers = nn.ModuleList()
        input_channels = 3
        for channels, stride in zip(self.channels, STRIDES[depth]):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, channels, 3, stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            input_channels = channels


class ResNet18(LayerwiseEncoder):
    # cifar resnet-18 (he et al., 2016) with every channel count multiplied by width
    # the stem is a stride-1 3x3 convolution with no max pooling, as in simclr's cifar runs
    # (chen et al., 2020), and greedy training treats each of the four stages as one layer,
    # like the blockwise training of siddiqui et al. (2024); the first layer also holds the stem

    def __init__(self, width):
        super().__init__()
        self.depth = len(RESNET18_CHANNELS)
        self.channels = [round(c * width) for c in RESNET18_CHANNELS]
        input_channels = self.channels[0]
        stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList()
        for index, (channels, stride) in enumerate(zip(self.channels, RESNET18_STRIDES)):
            # two residual blocks per stage; only the first one changes the shape
            blocks = []
            for block_stride in (stride, 1):
                # a 1x1 convolution matches the shortcut to the new shape when needed
                shortcut = None
                if block_stride != 1 or input_channels != channels:
                    shortcut = nn.Sequential(
                        nn.Conv2d(input_channels, channels, 1, block_stride, bias=False),
                        nn.BatchNorm2d(channels),
                    )
                blocks.append(BasicBlock(input_channels, channels, block_stride, shortcut))
                input_channels = channels
            stage = nn.Sequential(*blocks)
            self.layers.append(nn.Sequential(stem, stage) if index == 0 else stage)
        # use torchvision's resnet initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


def base_channels(settings):
    # each layer's channel count at width 1
    if settings.get("architecture", "convnet") == "resnet18":
        return RESNET18_CHANNELS
    return CHANNELS[settings["depth"]]


def build_encoder(settings, width):
    # older checkpoints don't save an architecture; those are all convolutional encoders
    if settings.get("architecture", "convnet") == "resnet18":
        return ResNet18(width)
    return Encoder(settings["depth"], width)


def make_encoder(settings, device):
    # seed the weights separately and restore the cpu random state afterward
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed("encoder", settings))
        encoder = build_encoder(settings, settings["width"])
    encoder = encoder.to(device)
    # under mup, mark which weight dimensions grow with width (does nothing otherwise)
    set_width_shapes(encoder, lambda width: build_encoder(settings, width), settings)
    return encoder


def load_encoder(checkpoint, device):
    # rebuild a saved encoder from its run settings and weights
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = checkpoint["configuration"]
    encoder = build_encoder(settings, settings["width"])
    encoder.load_state_dict(checkpoint["state_dict"])
    return encoder.to(device).eval(), settings
