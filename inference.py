import argparse

import torch
import torch.nn.functional as F

from dataset import RobotDataset, collate_fn
from logger import load_checkpoint


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(network, dataloader, device, target_names):
    network.eval()
    total_loss = 0.0
    absolute_error = torch.zeros(len(target_names), dtype=torch.float64)
    squared_error = torch.zeros(len(target_names), dtype=torch.float64)
    count = 0

    for batch in dataloader:
        batch = to_device(batch, device)
        targets = batch["targets"]
        with autocast(device):
            predictions = network(**batch)
        loss = F.smooth_l1_loss(predictions.float(), targets.float())
        errors = predictions.float() - targets.float()
        total_loss += loss.item() * len(targets)
        absolute_error += errors.abs().sum(dim=0).cpu()
        squared_error += errors.square().sum(dim=0).cpu()
        count += len(targets)

    metrics = {"loss": total_loss / count}
    for index, name in enumerate(target_names):
        metrics[f"mae_{name}"] = (absolute_error[index] / count).item()
        metrics[f"rmse_{name}"] = (squared_error[index] / count).sqrt().item()
    network.train()
    return metrics


@torch.no_grad()
def predict_outcomes(network, meta, image_path, state, action, goal, device):
    """Predict configured future outcomes for one image/state/action tuple."""
    example = {
        "image_path": str(image_path),
        "goal": goal,
        "target_names": meta["target_names"],
        "state": torch.tensor(state, dtype=torch.float32),
        "action": torch.tensor(action, dtype=torch.float32),
        "prediction_horizon_steps": meta["prediction_horizon_steps"],
    }
    batch = to_device(collate_fn([example]), device)
    network.eval()
    with autocast(device):
        values = network(**batch)[0].float().cpu().tolist()
    return dict(zip(meta["target_names"], values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<run-id>/checkpoints/<name>")
    parser.add_argument("--dataset_path", help="saved Hugging Face DatasetDict directory")
    parser.add_argument("--split", help="dataset split to evaluate; defaults to validation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    network, meta = load_checkpoint(args.checkpoint, device)
    dataset_path = args.dataset_path or meta["dataset_path"]
    split = args.split or meta.get("val_split") or meta["train_split"]
    dataset = RobotDataset(dataset_path, split, meta["target_names"], meta["prediction_horizon_steps"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                         num_workers=args.num_workers)
    print(evaluate(network, loader, device, meta["target_names"]))


if __name__ == "__main__":
    main()
