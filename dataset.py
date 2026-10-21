import hashlib
from pathlib import Path

import torch
from datasets import Image as HFImage
from datasets import load_from_disk
from torch.utils.data import Dataset

from tokenization import processor


class RobotDataset(Dataset):
    """One Hugging Face DatasetDict split of image/state/action/future-target rows."""

    def __init__(self, dataset_path, split, target_names, prediction_horizon_steps):
        self.dataset_path = Path(dataset_path).resolve()
        self.split = split
        self.target_names = list(target_names)
        self.prediction_horizon_steps = prediction_horizon_steps
        self.dataset = load_from_disk(str(self.dataset_path))[split].cast_column(
            "image", HFImage(decode=False)
        )
        if not len(self.dataset):
            raise ValueError(f"No examples found in split {split!r} at {self.dataset_path}")

        columns = set(self.dataset.column_names)
        required = {"image", "goal", "state", "action", *self.target_names}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Split {split!r} is missing required columns: {missing}")

        first = self.dataset[0]
        self.state_dim = len(first["state"])
        self.action_dim = len(first["action"])
        self.image_cache = self.dataset_path / ".materialized_images"

    def __len__(self):
        return len(self.dataset)

    def _image_path(self, image, index):
        path = image.get("path")
        if path:
            path = Path(path)
            if not path.is_absolute():
                path = self.dataset_path / path
            if path.is_file():
                return path

        data = image.get("bytes")
        if not data:
            raise FileNotFoundError(f"Image bytes/path missing for {self.split} row {index}")
        self.image_cache.mkdir(parents=True, exist_ok=True)
        suffix = hashlib.sha256(data).hexdigest() + ".jpg"
        output_path = self.image_cache / suffix
        if not output_path.exists():
            output_path.write_bytes(data)
        return output_path

    def __getitem__(self, index):
        row = self.dataset[index]
        state = [float(value) for value in row["state"]]
        action = [float(value) for value in row["action"]]
        targets = [float(row[name]) for name in self.target_names]
        if len(state) != self.state_dim or len(action) != self.action_dim:
            raise ValueError(f"Inconsistent state/action dimensions in split {self.split!r}, row {index}")

        return {
            "image_path": str(self._image_path(row["image"], index)),
            "goal": str(row["goal"]),
            "target_names": self.target_names,
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "prediction_horizon_steps": self.prediction_horizon_steps,
        }


def collate_fn(batch):
    conversations = []
    for example in batch:
        action_text = ", ".join(f"{value:.6g}" for value in example["action"].tolist())
        prompt = (
            f"Goal: {example['goal'] or 'No goal specified.'}\n"
            f"Candidate action values: [{action_text}]\n"
            f"Predict these numeric targets {example['prediction_horizon_steps']} control steps after this action: "
            f"{', '.join(example['target_names'])}."
        )
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image", "url": example["image_path"]},
                {"type": "text", "text": prompt},
            ],
        }])

    model_inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    )
    out = dict(model_inputs)
    out["robot_state"] = torch.stack([example["state"] for example in batch])
    out["robot_action"] = torch.stack([example["action"] for example in batch])
    if all("targets" in example for example in batch):
        out["targets"] = torch.stack([example["targets"] for example in batch])
    return out
