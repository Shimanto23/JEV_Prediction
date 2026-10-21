import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText


def text_config(config):
    return getattr(config, "text_config", config)


def load_backbone(model_name):
    return AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.float32)


class OutcomeHead(nn.Module):
    """Fuse image-text context and structured robot state/action features into outcome values."""

    def __init__(self, hidden_dim, state_dim, action_dim, target_dim, dropout=0.1):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, image_text, robot_state, robot_action):
        fused = torch.cat([
            image_text,
            self.state_encoder(robot_state),
            self.action_encoder(robot_action),
        ], dim=-1)
        return self.fusion(fused)


class RobotPredictor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask, robot_state, robot_action,
                targets=None, **vision_inputs):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **vision_inputs,
        )
        hidden = output.hidden_states[-1].float()
        token_positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
        last_token = token_positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        image_text = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token]
        robot_state = robot_state.to(dtype=image_text.dtype)
        robot_action = robot_action.to(dtype=image_text.dtype)
        # This head predicts continuous future state values in the dataset's units
        # (joint radians and gripper position), so its output must remain unbounded.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(image_text, robot_state, robot_action)


def build_network(model_name, lora_enabled, lora_r, lora_alpha, lora_dropout,
                  state_dim, action_dim, target_dim, head_dropout=0.1):
    backbone = load_backbone(model_name)
    if lora_enabled:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        )
        backbone = get_peft_model(backbone, lora_config)
    else:
        backbone.requires_grad_(False)
    hidden_dim = text_config(backbone.config).hidden_size
    head = OutcomeHead(hidden_dim, state_dim, action_dim, target_dim, head_dropout)
    return RobotPredictor(backbone, head)
