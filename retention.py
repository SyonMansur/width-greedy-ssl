# how much of each layer's features survive into the next layer, measured without class labels
# a ridge regression learns to recover layer l's pooled features from layer l + 1's (a learned
# linear map between two representations, as in lenc & vedaldi, 2015). the penalty is chosen on
# the linear probe's held-out images, the map is refit on all training images, and we report
#   r^2 = 1 - sum ||y - y_hat||^2 / sum ||y - mean(y)||^2   on the test images

import csv
from pathlib import Path

import numpy as np
import torch

from architecture import load_encoder
from linear_probe import encoder_features, validation_split


# set these paths for an existing checkpoint, then run python retention.py
CHECKPOINT = "results/barlow/final_encoder.pt"
DATA_DIR = "data"
OUTPUT = "results/retention.csv"

# penalties as multiples of the predictor features' average eigenvalue, from 1e-8 to 1e2
ALPHAS = 10.0 ** np.arange(-8.0, 2.01, 0.5)


class Ridge:
    # ridge regression with an intercept from one layer's features, solved for any penalty
    # through one eigendecomposition: W = V diag(1 / (s + lambda)) V^T Xc^T Yc

    def __init__(self, x):
        self.mean = x.mean(0)
        centered = x - self.mean
        eigenvalues, self.vectors = torch.linalg.eigh(centered.T @ centered)
        self.eigenvalues = eigenvalues.clamp_min(0)
        self.average_eigenvalue = self.eigenvalues.sum() / x.shape[1]
        self.rotated = centered @ self.vectors

    def r2(self, y, x_test, y_test, alpha):
        # fit the map to y, then score its predictions for held-out images
        cross = self.rotated.T @ (y - y.mean(0))
        shrink = 1 / (self.eigenvalues + alpha * self.average_eigenvalue)
        predicted = ((x_test - self.mean) @ self.vectors * shrink) @ cross + y.mean(0)
        error = (y_test - predicted).square().sum()
        return (1 - error / (y_test - y_test.mean(0)).square().sum()).item()


if __name__ == "__main__":
    # full float32 precision, so the features don't depend on tensor-core rounding
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, settings = load_encoder(CHECKPOINT, device)
    dataset = settings["dataset"].lower().replace("-", "")
    train, train_labels = encoder_features(encoder, dataset, True, dataset, DATA_DIR, device)
    test, _ = encoder_features(encoder, dataset, False, dataset, DATA_DIR, device)
    # the labels are only used to hold out the same images as the linear probe
    fit, validation = validation_split(train_labels)
    rows = []
    for layer in range(1, encoder.depth):
        # predict this layer's features (y) from the next layer's (x)
        x, y = train[layer].to(device), train[layer - 1].to(device)
        x_test, y_test = test[layer].to(device), test[layer - 1].to(device)
        # choose the penalty on the held-out images, then refit on all training images
        held_out = Ridge(x[fit])
        scores = [held_out.r2(y[fit], x[validation], y[validation], alpha) for alpha in ALPHAS]
        alpha = ALPHAS[int(np.argmax(scores))]
        r2 = Ridge(x).r2(y, x_test, y_test, alpha)
        rows.append({"layer": layer, "from_layer": layer + 1, "alpha": float(alpha), "test_r2": r2})
        print(f"Layer {layer} from layer {layer + 1}: R^2 {r2:.3f}", flush=True)
    # write one row per layer pair and leave any existing output file alone
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with Path(OUTPUT).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
