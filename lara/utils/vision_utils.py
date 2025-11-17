import base64
from io import BytesIO
from typing import Optional
import numpy as np
import torch
from PIL import Image


def encode_image(rgb: np.ndarray) -> str:
    """
    Encode RGB numpy array as base64 string for API.

    Args:
        rgb: RGB image as numpy array, shape (3, H, W) or (H, W, 3)

    Returns:
        Base64 encoded image string
    """
    # Handle different input shapes
    if rgb.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        rgb = np.transpose(rgb, (1, 2, 0))

    # Convert to uint8 if needed
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(rgb)

    # Encode as PNG
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Base64 encode
    image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_b64


def format_state_action_history(
    states,
    actions,
    state_is_pad: Optional[torch.Tensor] = None,
    action_is_pad: Optional[torch.Tensor] = None,
) -> str:
    """
    Format proprioceptive state and action history as text.

    Args:
        states: State history as dict of tensors or tensor, shape (T, state_dim)
                If dict, keys are body parts (e.g., 'torso', 'left_arm')
        actions: Action history as dict of tensors or tensor, shape (T, action_dim)
                If dict, keys are action types (e.g., 'base', 'torso')
        state_is_pad: Padding mask for states, shape (T,)
        action_is_pad: Padding mask for actions, shape (T,)

    Returns:
        Formatted string describing state/action history
    """
    history_text = "Recent State-Action History:\n"

    # Handle dict or tensor states/actions
    if isinstance(states, dict):
        # States and actions are dictionaries of tensors
        # Get time dimension from first tensor in states
        first_key = next(iter(states.keys()))
        T = states[first_key].shape[0]

        for t in range(T):
            # Check if padded
            if state_is_pad is not None and state_is_pad[t]:
                continue

            state_str = f"  State[t-{T-t-1}]: "
            # Format each body part
            for key, tensor in states.items():
                tensor_np = tensor[t].cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor[t]
                # Show first 4 values
                values_str = np.array2string(tensor_np.flatten()[:4], precision=4, separator=", ")
                state_str += f"{key}={values_str}... "

            action_str = f"  Action[t-{T-t-1}]: "
            # Format each action type
            if isinstance(actions, dict):
                for key, tensor in actions.items():
                    tensor_np = tensor[t].cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor[t]
                    # Show first 4 values
                    values_str = np.array2string(tensor_np.flatten()[:4], precision=4, separator=", ")
                    action_str += f"{key}={values_str}... "
            else:
                # Fallback if actions is tensor
                actions_np = actions[t].cpu().numpy() if isinstance(actions, torch.Tensor) else actions[t]
                action_str += f"{actions_np[:8]}..."

            history_text += f"{state_str}\n{action_str}\n"
    else:
        # States and actions are tensors
        states = states.cpu().numpy()
        actions = actions.cpu().numpy()
        T = states.shape[0]
        for t in range(T):
            # Check if padded
            if state_is_pad is not None and state_is_pad[t]:
                continue

            state_str = f"  State[t-{T-t-1}]: {states[t][:8]}..."  # Show first 8 dims
            action_str = f"  Action[t-{T-t-1}]: {actions[t][:8]}..."
            history_text += f"{state_str}\n{action_str}\n"

    return history_text
