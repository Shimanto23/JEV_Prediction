import os

from transformers import AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
