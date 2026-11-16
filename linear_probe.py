# linear probe on a saved encoder's frozen features, following kornblith, shlens & le (2019)
# a multinomial logistic regression predicts the class from each image's pooled features, with
# no augmentation and an l2 penalty lambda * ||W||^2 on the summed cross-entropy (the bias isn't
# penalized). lambda is chosen from 45 log-spaced values between 1e-6 and 1e5 by accuracy on 10%
# of the training images, held out evenly across classes; the chosen probe is then refit on all
# training images and scored once on the test images

import csv
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from architecture import load_encoder
from data import CIFAR


# set these paths for an existing checkpoint, then run python linear_probe.py
CHECKPOINT = "results/barlow/final_encoder.pt"
DATASET = "cifar10"  # "cifar10", or "cifar100" to test how a cifar-10 encoder transfers
LAYERS = "final"  # "final" for the last layer only, or "all"
DATA_DIR = "data"
OUTPUT = "results/linear_probe.csv"

# try the strongest penalty first, then start each weaker one from the previous solution
LAMBDAS = np.logspace(-6, 5, 45)[::-1]
SPLIT_SEED = 2026  # the validation split used by every hyperparameter search in the paper
GRADIENT_TOLERANCE = 1e-4  # stop once every gradient entry is smaller, as in scikit-learn
CURVATURE = 0.1  # rough softmax curvature used to rescale the weights; only changes the speed


@torch.inference_mode()
def encoder_features(encoder, dataset, train, encoder_dataset, data_dir, device):
    # pooled features from every layer for the unaugmented images, in the dataset's order
    # normalize with the statistics of the encoder's own training data, even for transfer
    dataset_class = CIFAR[dataset][0]
    _, mean, std = CIFAR[encoder_dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    images = dataset_class(data_dir, train=train, download=True, transform=transform)
    features, labels = [[] for _ in encoder.layers], []
    for batch, batch_labels in DataLoader(images, batch_size=500, num_workers=4):
        for index, pooled in enumerate(encoder.layer_features(batch.to(device))):
            features[index].append(pooled.double().cpu())
        labels.append(batch_labels)
    return [torch.cat(layer) for layer in features], torch.cat(labels)


def validation_split(labels, seed=SPLIT_SEED):
    # hold out 10% of each class's training images to choose lambda
    # (500 per class for cifar-10, 50 per class for cifar-100)
    labels = np.asarray(labels)
    classes = labels.max() + 1
    held_out = len(labels) // 10 // classes
    generator = np.random.default_rng(seed)
    fit, validation = [], []
    for label in range(classes):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        validation.extend(indices[:held_out])
        fit.extend(indices[held_out:])
    generator.shuffle(fit)
    generator.shuffle(validation)
    return np.array(fit), np.array(validation)


class Basis:
    # coordinates where l-bfgs converges quickly, computed once from the training features:
    # center the features, then rotate them onto the eigenvectors of their gram matrix
    # neither step changes the best probe: the unpenalized bias absorbs the centering, and the
    # rotation keeps ||W|| the same

    def __init__(self, features):
        self.mean = features.mean(0)
        centered = features - self.mean
        eigenvalues, self.vectors = torch.linalg.eigh(centered.T @ centered)
        self.eigenvalues = eigenvalues.clamp_min(0)

    def rotate(self, features):
        return (features - self.mean) @ self.vectors


def fit_probe(z, labels, lambd, start, basis):
    # minimize sum_i cross_entropy(z_i W + b, y_i) + lambd * ||W||^2 for rotated features z,
    # starting from an earlier (W, b)
    # l-bfgs sees rescaled weights, W = r * u with one scale per eigenvector and b = r_b * beta,
    # which gives every coordinate about the same curvature; this only changes the speed
    start_weights, start_bias = start
    scale = torch.rsqrt(CURVATURE * basis.eigenvalues + 2 * lambd)[:, None]
    bias_scale = (CURVATURE * len(z)) ** -0.5
    targets = F.one_hot(labels, start_weights.shape[1]).double()
    size = start_weights.numel()
    latest = {}

    def loss_and_gradient(vector):
        # unpack l-bfgs's flat vector into the weights and bias
        weights = scale * torch.from_numpy(vector[:size]).to(z.device).view_as(start_weights)
        bias = bias_scale * torch.from_numpy(vector[size:]).to(z.device)
        logits = z @ weights + bias
        cross_entropy = torch.logsumexp(logits, 1) - (logits * targets).sum(1)
        loss = cross_entropy.sum() + lambd * weights.square().sum()
        # the cross-entropy's gradient at the logits is the softmax minus the one-hot labels
        residual = logits.softmax(1) - targets
        weight_gradient = z.T @ residual + 2 * lambd * weights
        bias_gradient = residual.sum(0)
        latest.update(vector=vector.copy(), weights=weights, bias=bias)
        latest.update(weight_gradient=weight_gradient, bias_gradient=bias_gradient)
        gradient = torch.cat([(scale * weight_gradient).flatten(), bias_scale * bias_gradient])
        return loss.item(), gradient.cpu().numpy()

    def largest_gradient():
        # largest gradient entry of the same probe written for the original, uncentered features
        weight_gradient = basis.vectors @ latest["weight_gradient"]
        weight_gradient += torch.outer(basis.mean, latest["bias_gradient"])
        return max(weight_gradient.abs().max().item(), latest["bias_gradient"].abs().max().item())

    def stop_when_converged(intermediate_result):
        if not np.array_equal(intermediate_result.x, latest["vector"]):
            loss_and_gradient(intermediate_result.x)
        if largest_gradient() < GRADIENT_TOLERANCE:
            raise StopIteration

    vector = torch.cat([(start_weights / scale).flatten(), start_bias / bias_scale])
    vector = vector.cpu().numpy()
    loss_and_gradient(vector)
    if largest_gradient() >= GRADIENT_TOLERANCE:
        # scikit-learn's l-bfgs settings, which also stop once the loss stops decreasing
        result = scipy.optimize.minimize(
            loss_and_gradient,
            vector,
            jac=True,
            method="L-BFGS-B",
            callback=stop_when_converged,
            options={
                "maxiter": 30000,
                "maxcor": 10,
                "maxls": 50,
                "gtol": 0.0,
                "ftol": 64 * np.finfo(float).eps,
            },
        )
        if not np.array_equal(result.x, latest["vector"]):
            loss_and_gradient(result.x)
    return latest["weights"].clone(), latest["bias"].clone()


def accuracy(basis, probe, features, labels):
    weights, bias = probe
    predictions = (basis.rotate(features) @ weights + bias).argmax(1)
    return 100.0 * (predictions == labels).double().mean().item()


def linear_probe(train_features, train_labels, test_features, test_labels):
    # the full protocol for one layer: choose lambda on the held-out images, refit, then test
    fit, validation = validation_split(train_labels.cpu())
    classes = int(train_labels.max()) + 1
    basis = Basis(train_features[fit])
    z = basis.rotate(train_features[fit])
    probe = (
        torch.zeros(z.shape[1], classes, dtype=torch.float64, device=z.device),
        torch.zeros(classes, dtype=torch.float64, device=z.device),
    )
    probes, validation_accuracy = [], []
    for lambd in LAMBDAS:
        probe = fit_probe(z, train_labels[fit], lambd, probe, basis)
        probes.append(probe)
        validation_accuracy.append(
            accuracy(basis, probe, train_features[validation], train_labels[validation])
        )
    # pick the best validation accuracy; ties go to the stronger penalty
    best = max(range(len(LAMBDAS)), key=lambda i: (validation_accuracy[i], -i))
    # refit on all the training images, starting from the chosen probe
    z = basis.rotate(train_features)
    probe = fit_probe(z, train_labels, LAMBDAS[best], probes[best], basis)
    return {
        "lambda": float(LAMBDAS[best]),
        "validation_accuracy": validation_accuracy[best],
        "test_accuracy": accuracy(basis, probe, test_features, test_labels),
    }


if __name__ == "__main__":
    # full float32 precision, so the features don't depend on tensor-core rounding
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, settings = load_encoder(CHECKPOINT, device)
    # accept the dataset spelling used by both old and new checkpoints
    encoder_dataset = settings["dataset"].lower().replace("-", "")
    train_features, train_labels = encoder_features(
        encoder, DATASET, True, encoder_dataset, DATA_DIR, device
    )
    test_features, test_labels = encoder_features(
        encoder, DATASET, False, encoder_dataset, DATA_DIR, device
    )
    layers = range(1, encoder.depth + 1) if LAYERS == "all" else [encoder.depth]
    rows = []
    for layer in layers:
        result = linear_probe(
            train_features[layer - 1].to(device),
            train_labels.to(device),
            test_features[layer - 1].to(device),
            test_labels.to(device),
        )
        rows.append({"dataset": DATASET, "layer": layer, **result})
        print(f"Layer {layer}: linear probe {result['test_accuracy']:.2f}%", flush=True)
    # write one row per layer and leave any existing output file alone
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with Path(OUTPUT).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
