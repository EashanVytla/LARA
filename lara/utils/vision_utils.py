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
    states: torch.Tensor,
    actions: torch.Tensor,
    state_is_pad: Optional[torch.Tensor] = None,
    action_is_pad: Optional[torch.Tensor] = None,
) -> str:
    """
    Format proprioceptive state and action history as text.

    Args:
        states: State history tensor, shape (T, state_dim)
        actions: Action history tensor, shape (T, action_dim)
        state_is_pad: Padding mask for states, shape (T,)
        action_is_pad: Padding mask for actions, shape (T,)

    Returns:
        Formatted string describing state/action history
    """
    # Convert to numpy
    states = states.cpu().numpy()
    actions = actions.cpu().numpy()

    history_text = "Recent State-Action History:\n"

    T = states.shape[0]
    for t in range(T):
        # Check if padded
        if state_is_pad is not None and state_is_pad[t]:
            continue

        state_str = f"  State[t-{T-t-1}]: {states[t][:8]}..."  # Show first 8 dims
        action_str = f"  Action[t-{T-t-1}]: {actions[t][:8]}..."
        history_text += f"{state_str}\n{action_str}\n"

    return history_text
