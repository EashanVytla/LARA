import os
import base64
from io import BytesIO
from typing import Optional, Dict, Any
import numpy as np
import torch
from anthropic import Anthropic
from PIL import Image


class LLMInterface:
    """Interface for Claude API to generate subtask ground truth from robot observations."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        """
        Initialize the LLM interface with Claude API.

        Args:
            model: Claude model to use (e.g., "claude-3-haiku-20240307")
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            api_key_env: Environment variable name containing API key
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                f"API key not found. Please set {api_key_env} in your .env file. "
                f"Get your API key from: https://console.anthropic.com/"
            )

        self.client = Anthropic(api_key=api_key)

    def _encode_image(self, rgb: np.ndarray) -> str:
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

    def _format_state_action_history(
        self,
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

    def _detect_stuck(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        velocity_threshold: float = 0.01,
    ) -> bool:
        """
        Detect if the robot appears to be stuck.

        Args:
            states: State history tensor, shape (T, state_dim)
            actions: Action history tensor, shape (T, action_dim)
            velocity_threshold: Threshold for zero velocity detection

        Returns:
            True if robot appears stuck
        """
        # Check if actions are non-zero but states barely changed
        action_norm = torch.norm(actions[-1])

        if action_norm > 0.1:  # Robot is trying to act
            # Check if position/velocity changed
            if states.shape[0] >= 2:
                state_diff = torch.norm(states[-1] - states[-2])
                if state_diff < velocity_threshold:
                    return True

        return False

    def _build_prompt(
        self,
        task: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_stuck_detection: bool = True,
        velocity_threshold: float = 0.01,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Build the prompt for subtask generation.

        Args:
            task: High-level task description
            states: State history tensor
            actions: Action history tensor
            use_stuck_detection: Whether to include stuck detection
            velocity_threshold: Threshold for stuck detection
            state_is_pad: Padding mask for states
            action_is_pad: Padding mask for actions

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing a robot performing the following task:
Task: {task}

You are given:
1. The current RGB camera view from the robot's perspective
2. The robot's recent proprioceptive state history (joint positions, velocities, etc.)
3. The robot's recent action history

Your goal is to describe the IMMEDIATE subtask the robot should be performing at this moment.

{self._format_state_action_history(states, actions, state_is_pad, action_is_pad)}

"""

        if use_stuck_detection and self._detect_stuck(states, actions, velocity_threshold):
            prompt += """
⚠️ IMPORTANT: The robot appears to be stuck! The actions show non-zero commands but the state
indicates minimal movement. The subtask should address this obstruction.
"""

        prompt += """
Guidelines:
- The robot may not see the target object in frame, so it may need to explore first
- Break down complex tasks into atomic subtasks (e.g., "Turn 360 degrees to find the trash can")
- If stuck (velocity near zero despite acting), suggest strategies to work around obstruction
- Be specific and actionable (e.g., "Rotate left 45 degrees" not "Look around")
- Focus on what the robot should do RIGHT NOW, not the entire task plan
- Keep response under 2 sentences

Subtask description:"""

        return prompt

    def predict_subtask(
        self,
        rgb_images: dict[str, np.ndarray],
        task: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_stuck_detection: bool = True,
        velocity_threshold: float = 0.01,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate subtask ground truth using Claude API.

        Args:
            rgb_images: Dictionary of RGB images {camera_name: image}, each shape (3, H, W) or (H, W, 3)
            task: High-level task description
            states: State history tensor, shape (T, state_dim)
            actions: Action history tensor, shape (T, action_dim)
            use_stuck_detection: Whether to detect and handle stuck situations
            velocity_threshold: Threshold for stuck detection
            state_is_pad: Optional padding mask for states
            action_is_pad: Optional padding mask for actions

        Returns:
            Natural language subtask description
        """
        # Encode all images
        encoded_images = {cam: self._encode_image(rgb) for cam, rgb in rgb_images.items()}

        # Build prompt
        prompt = self._build_prompt(
            task=task,
            states=states,
            actions=actions,
            use_stuck_detection=use_stuck_detection,
            velocity_threshold=velocity_threshold,
            state_is_pad=state_is_pad,
            action_is_pad=action_is_pad,
        )

        # Build content with multiple images
        content = []
        for image_b64 in encoded_images.values():
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })
        content.append({
            "type": "text",
            "text": f"Camera views: {', '.join(encoded_images.keys())}\n\n{prompt}"
        })

        # Call Claude API
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )

            # Extract response
            subtask = message.content[0].text.strip()
            return subtask

        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {str(e)}")
