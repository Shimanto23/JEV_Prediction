"""Build a column-based Hugging Face DatasetDict from a small LeRobot dataset.

Install the LeRobot dataset decoder on demand with:
  uv run --isolated --no-project --python 3.12 --with 'lerobot[dataset]==0.6.1' python prepare_hf_robot_data.py
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value
from PIL import Image


DEFAULT_DATASET = "Bigenlight/cube_in_cup_lerobot_v3"
TARGET_NAMES = [f"future_{name}" for name in (
    "ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos",
)]


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_frame(value, image_path):
    array = to_numpy(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="JPEG", quality=90, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face LeRobot dataset ID")
    parser.add_argument("--output", default="data/cube_in_cup", help="Output DatasetDict directory")
    parser.add_argument("--horizon", type=int, default=1, help="Future frame offset within each episode")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is required to decode the Hub dataset. Run this script with "
            "`uv run --isolated --no-project --python 3.12 --with 'lerobot[dataset]==0.6.1' "
            "python prepare_hf_robot_data.py`."
        ) from exc

    print(f"Loading {args.dataset} from Hugging Face (first run downloads the dataset)...")
    source = LeRobotDataset(args.dataset)
    output_root = Path(args.output).resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    # Each row follows a column-oriented schema, with an actual Image feature.
    states, actions, episodes, frames = [], [], [], []
    for index in range(len(source)):
        sample = source[index]
        episode = int(to_numpy(sample["episode_index"]).reshape(-1)[0])
        frame = int(to_numpy(sample["frame_index"]).reshape(-1)[0])
        state = to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)
        action = to_numpy(sample["action"]).astype(np.float32).reshape(-1)
        if state.shape != (7,) or action.shape != (7,):
            raise ValueError(f"Expected 7-D state/action, got {state.shape}/{action.shape} at row {index}")

        image_path = images_root / f"episode_{episode:05d}" / f"frame_{frame:06d}.jpg"
        if not image_path.exists():
            save_frame(sample["observation.images.cam1"], image_path)
        states.append(state)
        actions.append(action)
        episodes.append(episode)
        frames.append(frame)
        if (index + 1) % 500 == 0 or index + 1 == len(source):
            print(f"Decoded {index + 1}/{len(source)} frames")

    states = np.stack(states)
    actions = np.stack(actions)
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    episode_ids = sorted(set(episodes.tolist()))
    if len(episode_ids) < 2:
        raise ValueError("At least two episodes are needed for an episode-level train/validation split")
    random.Random(args.seed).shuffle(episode_ids)
    validation_count = max(1, round(len(episode_ids) * args.validation_fraction))
    validation_episodes = set(episode_ids[:validation_count])

    rows = {"train": [], "validation": []}
    by_episode = {}
    for index, episode in enumerate(episodes):
        by_episode.setdefault(int(episode), []).append(index)
    for episode, indexes in by_episode.items():
        indexes.sort(key=lambda i: frames[i])
        split = "validation" if episode in validation_episodes else "train"
        for position in range(len(indexes) - args.horizon):
            current = indexes[position]
            future = indexes[position + args.horizon]
            if frames[future] - frames[current] != args.horizon:
                continue
            frame = int(frames[current])
            row = {
                "image": str(images_root / f"episode_{episode:05d}" / f"frame_{frame:06d}.jpg"),
                "goal": "Put the cube in the cup.",
                "state": states[current].tolist(),
                "action": actions[current].tolist(),
                "episode_index": episode,
                "frame_index": frame,
            }
            row.update(dict(zip(TARGET_NAMES, states[future].tolist())))
            rows[split].append(row)

    features = Features({
        "image": HFImage(),
        "goal": Value("string"),
        "state": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
        **{name: Value("float32") for name in TARGET_NAMES},
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
    })
    prepared = DatasetDict({
        split: Dataset.from_list(examples, features=features)
        for split, examples in rows.items()
    })
    prepared.save_to_disk(str(output_root))

    print(f"Saved Hugging Face DatasetDict -> {output_root}")
    for split, examples in prepared.items():
        print(f"{split}: {len(examples)} rows; columns={examples.column_names}")
    print(f"Episode-level validation episodes: {sorted(validation_episodes)}")
    print(f"Prediction target: 7-D robot state {args.horizon} frames in the future (30 fps dataset)")


if __name__ == "__main__":
    main()
