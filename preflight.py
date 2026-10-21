"""Run a single image/state/action forward pass before a training run."""

import argparse
from pathlib import Path

import torch
import yaml

from dataset import RobotDataset, collate_fn
from inference import autocast, get_device, to_device
from network import build_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML training configuration")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data = config["data"]
    dataset = RobotDataset(
        data.get("path", "data/robot"),
        data.get("train_split", "train"),
        data["target_names"],
        data["prediction_horizon_steps"],
    )
    device = get_device()
    batch = to_device(collate_fn([dataset[0]]), device)
    model = config["model"]
    lora = config["lora"]
    network = build_network(
        model["name"], lora["enabled"], lora["r"], lora["alpha"], lora["dropout"],
        dataset.state_dim, dataset.action_dim, len(data["target_names"]),
        config["head"].get("dropout", 0.1),
    ).to(device).eval()

    with torch.inference_mode(), autocast(device):
        predictions = network(**batch)
    expected = batch["targets"].shape
    if predictions.shape != expected:
        raise RuntimeError(f"Output shape {tuple(predictions.shape)} != target shape {tuple(expected)}")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Model produced NaN or infinite prediction values")

    print(f"Preflight passed: {Path(data.get('path', 'data/robot')).resolve()}")
    print(f"device={device}; model={model['name']}; output_shape={tuple(predictions.shape)}")
    print(f"state_dim={dataset.state_dim}; action_dim={dataset.action_dim}; targets={len(data['target_names'])}")


if __name__ == "__main__":
    main()
