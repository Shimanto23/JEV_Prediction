import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from dataset import RobotDataset, collate_fn
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger
from network import build_network


def make_loader(config, dataset, device, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        persistent_workers=config["data"]["num_workers"] > 0,
        pin_memory=device.type == "cuda",
    )


def resolve_checkpoint(resume):
    path = Path(resume)
    if not (path / "meta.json").exists():
        path = Path("runs") / resume / "checkpoints" / "final"
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no checkpoint at {resume} or {path}")
    return path


def load_weights(network, ckpt_dir):
    has_lora = (ckpt_dir / "lora").exists()
    if has_lora != isinstance(network.backbone, PeftModel):
        raise ValueError(f"{ckpt_dir} LoRA configuration does not match the training config")
    if has_lora:
        weights = load_file(ckpt_dir / "lora" / "adapter_model.safetensors")
        result = set_peft_model_state_dict(network.backbone, weights)
        missing = [key for key in result.missing_keys if "lora_" in key]
        if result.unexpected_keys or missing:
            raise ValueError(f"LoRA weights don't match: unexpected={result.unexpected_keys[:3]} missing={missing[:3]}")
    network.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))


def validate(network, loader, device, target_names, logger, step, best_loss):
    metrics = evaluate(network, loader, device, target_names)
    logger.log(step, split="validation", **metrics)
    if metrics["loss"] < best_loss:
        best_loss = metrics["loss"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_loss


def train(config, run_name, resume=None):
    torch.manual_seed(config["seed"])
    device = get_device()
    data, model, lora, optim, log = config["data"], config["model"], config["lora"], config["optim"], config["logging"]
    dataset_path = data.get("path", "data/robot")
    target_names = data["target_names"]
    horizon = data["prediction_horizon_steps"]

    train_dataset = RobotDataset(dataset_path, data.get("train_split", "train"), target_names, horizon)
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty")
    train_loader = make_loader(config, train_dataset, device, shuffle=True)
    val_dataset = None
    val_loader = None
    if data.get("val_split"):
        val_dataset = RobotDataset(dataset_path, data["val_split"], target_names, horizon)
        if (val_dataset.state_dim, val_dataset.action_dim) != (train_dataset.state_dim, train_dataset.action_dim):
            raise ValueError("Training and validation robot state/action dimensions must match")
        val_loader = make_loader(config, val_dataset, device, shuffle=False)

    config["data"]["state_dim"] = train_dataset.state_dim
    config["data"]["action_dim"] = train_dataset.action_dim
    resume_dir, resume_meta = None, {}
    if resume:
        resume_dir = resolve_checkpoint(resume)
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        config = {**config, "resumed_from": str(resume_dir)}

    logger = RunLogger(config, run_name, device)
    network = build_network(
        model["name"], lora["enabled"], lora["r"], lora["alpha"], lora["dropout"],
        train_dataset.state_dim, train_dataset.action_dim, len(target_names),
        config["head"].get("dropout", 0.1),
    )
    if resume_dir:
        load_weights(network, resume_dir)
        print(f"Loaded model weights from {resume_dir} (step {resume_meta['step']})")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    lora_params = [param for param in network.backbone.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    accum = optim.get("grad_accum_steps", 1)
    if accum < 1:
        raise ValueError("optim.grad_accum_steps must be at least 1")
    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    total_steps = optim["epochs"] * steps_per_epoch
    if total_steps < 1:
        raise ValueError("Training configuration produces no optimizer steps")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(optim["warmup_ratio"] * total_steps), total_steps,
    )

    step, best_loss = 0, float("inf")
    if resume_dir and val_loader is not None:
        _, best_loss = validate(network, val_loader, device, target_names, logger, step, best_loss)
    network.train()

    for epoch in range(optim["epochs"]):
        epoch_loss, epoch_count = 0.0, 0
        accum_loss, accum_count = 0.0, 0
        optimizer.zero_grad()
        for micro, batch in enumerate(train_loader, 1):
            group_start = ((micro - 1) // accum) * accum
            group_start_examples = group_start * data["batch_size"]
            group_examples = min(
                accum * data["batch_size"], len(train_dataset) - group_start_examples,
            )
            batch = to_device(batch, device)
            targets = batch["targets"]
            with autocast(device):
                predictions = network(**batch)
                per_example_loss = F.smooth_l1_loss(
                    predictions.float(), targets.float(), reduction="none",
                ).mean(dim=-1)
                loss = per_example_loss.mean()
            (per_example_loss.sum() / group_examples).backward()

            size = len(targets)
            epoch_loss += loss.item() * size
            epoch_count += size
            accum_loss += loss.item() * size
            accum_count += size
            if micro % accum and micro != len(train_loader):
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_lora"] = scheduler.get_last_lr()[0]
                logger.log(step, split="train", epoch=epoch, loss=accum_loss / accum_count, **lrs)
            accum_loss, accum_count = 0.0, 0

            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if val_loader is not None and step % log["eval_every"] == 0:
                _, best_loss = validate(network, val_loader, device, target_names, logger, step, best_loss)
                network.train()

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count)

    summary = {"final_train_loss": epoch_loss / epoch_count}
    if val_loader is not None:
        if step % log["eval_every"] != 0:
            val_metrics, best_loss = validate(network, val_loader, device, target_names, logger, step, best_loss)
        else:
            val_metrics = evaluate(network, val_loader, device, target_names)
        summary["validation"] = val_metrics
        summary["best_validation_loss"] = best_loss

    checkpoint = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(checkpoint), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config describing robot data and training")
    parser.add_argument("--name", help="run name (default: config file name)")
    parser.add_argument("--resume", help="initialize from a prior run/checkpoint; optimizer restarts")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume)


if __name__ == "__main__":
    main()
