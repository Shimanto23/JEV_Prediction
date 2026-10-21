# Robot next-state prediction with Qwen3.5 Vision

This project trains an action-conditioned robot state-transition predictor. Given a camera image, task goal, current numeric robot state, and recorded or candidate action, it directly regresses the robot state after a fixed horizon. It does not generate actions or future images. The regression output is produced in one forward pass, without autoregressive token generation; the Qwen3.5 backbone itself is still a causal decoder model.

The vision-language backbone is [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). The model uses its multimodal processor for image and goal/action text, plus a learned fusion head for numeric state and action values.

The included `arch.ipynb` demonstrates JEV-style choice isolation for Qwen3, but that packed-sequence attention-mask implementation is not used by the current Qwen3.5 trainer. Qwen3.5 has both full-attention and recurrent linear-attention layers ([Transformers architecture notes](https://huggingface.co/docs/transformers/model_doc/qwen3_5)), so the notebook mask cannot safely be assumed to isolate branches in every layer. Candidate actions are instead evaluated as independent batch rows by `predict_candidate_outcomes`; reordering the candidate list reorders predictions without letting candidates affect one another. This is a branch-isolated, JEV-inspired use of the predictor, not a direct port of the notebook's packed-sequence mechanism.

## Included Hugging Face robot dataset

The preparation script uses [`Bigenlight/cube_in_cup_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/cube_in_cup_lerobot_v3), a small Apache-2.0 LeRobot dataset with 23 UR7e teleoperation episodes, 6,177 frames, camera video, 7-D joint/gripper states, and 7-D actions. Its single task and pilot scale make it useful for getting the pipeline running, but not for claiming a robust robot predictor.

Prepare the data on the GPU machine. The first run downloads the source dataset and decodes its camera video:

```bash
uv run --isolated --no-project --python 3.12 --with 'lerobot[dataset]==0.6.1' python prepare_hf_robot_data.py
```

The script saves a Hugging Face `DatasetDict` under `data/cube_in_cup/` with `train` and `validation` splits. Its table has an `image` feature, `goal`, `state`, `action`, seven named future-state columns, and episode/frame IDs. The split is by episode to reduce leakage. Each input at frame `t` is paired with the observed 7-D robot state at `t + 1` frame (about 33 ms at 30 fps). Targets remain in robot units: joint radians and gripper position.

Before training, run a one-example preflight on the GPU host. This checks data loading, image processing, model loading, and the prediction output shape without updating weights:

```bash
uv run python preflight.py configs/hf_cube_in_cup_lora.yaml
```

Train with LoRA after preflight passes:

```bash
uv run python train.py configs/hf_cube_in_cup_lora.yaml --name cube-in-cup
```

## Custom robot datasets

Save a Hugging Face `DatasetDict` with `train` and `validation` splits. Use an `Image` feature for the camera frame and numeric columns for the robot inputs and future targets. For example, a row can have:

```text
image: Image
goal: string
state: list<float>
action: list<float>
future_ur_q1: float
future_ur_q2: float
...
episode_index: int
frame_index: int
```

The dataset viewer can render the image column. The model reads `state` and `action` from numeric columns and predicts the configured target columns. All rows need consistent state/action dimensions. Keep future measurements out of the input `state`; split by episode or scenario before training and validation. Edit `data.target_names` and `prediction_horizon_steps` in the config to match your columns.

Targets use Smooth L1 regression and can be continuous values in native units. Normalize target dimensions if they have very different scales, and keep the transform for deployment.

One row represents a demonstrated action and its later outcome. It does not provide counterfactual outcomes for other actions from that exact same state. At runtime, `predict_candidate_outcomes` evaluates a list of candidate actions independently for the same image/state, then a separate planner can apply its objective and safety constraints. Candidate actions unlike those represented in training data may not be reliable. For a JEV experiment that learns comparisons among multiple options at one state, collect or simulate outcome labels for multiple actions from that same state.

## Setup and training

Install [uv](https://docs.astral.sh/uv/) and the locked training dependencies:

```bash
uv sync
```

The first training run downloads the Qwen3.5-0.8B weights and processor. Training selects CUDA, Apple MPS, or CPU automatically. For an NVIDIA L40S, start with the provided small microbatches. `batch_size` counts dataset rows; `grad_accum_steps` controls the effective batch. Logs, validation metrics, and checkpoints are saved under `runs/<name>-<timestamp>/`.

The startup script is for a fresh Ubuntu GPU machine:

```bash
bash cloud_startup_script.sh
```

## Evaluate and predict

Evaluate a saved checkpoint on its validation split:

```bash
uv run python inference.py runs/<run-id>/checkpoints/best
```

It reports Smooth L1 loss, MAE, and RMSE for each target, plus a persistence baseline when target dimensions match the current state (predicting no state change). The Python APIs `inference.predict_outcomes(...)` and `inference.predict_candidate_outcomes(...)` predict one action outcome or a list of candidate outcomes for a shared observation.

## Project files

- [`dataset.py`](dataset.py): Hugging Face DatasetDict loading and multimodal batching.
- [`prepare_hf_robot_data.py`](prepare_hf_robot_data.py): downloads the public LeRobot dataset and creates the column-based DatasetDict.
- [`tokenization.py`](tokenization.py): Qwen3.5 processor setup.
- [`network.py`](network.py): image-text backbone, numeric feature fusion, and continuous prediction head.
- [`train.py`](train.py): training, gradient accumulation, validation, and checkpoint initialization.
- [`preflight.py`](preflight.py): one-example model/data compatibility check before training.
- [`logger.py`](logger.py): metrics, run files, and LoRA/head checkpoints.
- [`inference.py`](inference.py): evaluation and `predict_outcomes()`.
