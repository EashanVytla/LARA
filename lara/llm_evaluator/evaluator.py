import os
import re
from anthropic import Anthropic
import torch
from typing import Optional, Tuple
from omegaconf import DictConfig
from lara.utils import encode_image, format_state_action_history
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator for subtask predictions using Claude API."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key_env: str = "ANTHROPIC_API_KEY",
        prompts: Optional[DictConfig] = None,
    ):
        """
        Initialize the evaluator with Claude API.

        Args:
            model: Claude model to use (e.g., "claude-3-haiku-20240307")
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            api_key_env: Environment variable name containing API key
            prompts: Prompt configuration from prompts.yaml
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Store prompts if provided
        if prompts is not None:
            self.prompts = prompts
        else:
            logger.warning("Prompts set to None")
            self.prompts = None

        # Load API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                f"API key not found. Please set {api_key_env} in your .env file. "
                f"Get your API key from: https://console.anthropic.com/"
            )

        self.client = Anthropic(api_key=api_key)

    def _build_evaluation_prompt(
        self,
        subtask: str,
        task: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Build the prompt for evaluating a predicted subtask.

        Args:
            subtask: The predicted subtask to evaluate
            task: High-level task description
            states: State history tensor
            actions: Action history tensor
            state_is_pad: Padding mask for states
            action_is_pad: Padding mask for actions

        Returns:
            Formatted prompt string
        """
        if self.prompts is None:
            raise ValueError("Prompts config not provided to Evaluator")

        p = self.prompts  # Shorthand for prompt templates

        # Build evaluation prompt from configurable components
        prompt = f"{p.intro}\n"
        prompt += f"{p.task_line.format(task=task)}\n"
        prompt += f"{p.prediction_line.format(subtask=subtask)}\n\n"
        prompt += f"{p.context}\n\n"

        # Add state-action history
        prompt += format_state_action_history(states, actions, state_is_pad, action_is_pad)
        prompt += "\n\n"

        prompt += f"{p.goal}\n\n"
        prompt += f"{p.rating_instructions}\n\n"
        prompt += f"{p.output_format}"

        return prompt

    def _parse_evaluation(self, evaluation_text: str) -> Tuple[float, str]:
        """
        Parse the evaluation response to extract score and explanation.

        Args:
            evaluation_text: Raw text response from Claude

        Returns:
            Tuple of (score, explanation)

        Raises:
            ValueError: If the response cannot be parsed
        """
        lines = evaluation_text.strip().split('\n')

        score = None
        explanation = None

        # Parse score
        for line in lines:
            if line.strip().startswith('Score:'):
                score_text = line.split(':', 1)[1].strip()
                # Extract the first number found (handles cases like "0.8" or "0.8 out of 1.0")
                score_match = re.search(r'(\d+\.?\d*)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                break

        # Parse explanation
        for line in lines:
            if line.strip().startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()
                break

        # Validate parsed values
        if score is None:
            raise ValueError(f"Could not parse score from evaluation: {evaluation_text}")
        if explanation is None:
            raise ValueError(f"Could not parse explanation from evaluation: {evaluation_text}")

        # Clamp score to [0.0, 1.0] range
        score = max(0.0, min(1.0, score))

        return score, explanation

    def evaluate_prediction(
        self,
        subtask: str,
        rgb_images: dict[str, torch.Tensor],
        task: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_stuck_detection: bool = True,
        velocity_threshold: float = 0.01,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[float, str]:
        """
        Evaluate a predicted subtask using Claude API.

        Args:
            subtask: The predicted subtask to evaluate
            rgb_images: Dictionary of RGB images {camera_name: tensor}
            task: High-level task description
            states: State history tensor, shape (T, state_dim)
            actions: Action history tensor, shape (T, action_dim)
            use_stuck_detection: Whether to include stuck detection (currently unused)
            velocity_threshold: Threshold for stuck detection (currently unused)
            state_is_pad: Optional padding mask for states
            action_is_pad: Optional padding mask for actions

        Returns:
            Tuple of (score, explanation) where score is float in [0.0, 1.0]
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            subtask=subtask,
            task=task,
            states=states,
            actions=actions,
            state_is_pad=state_is_pad,
            action_is_pad=action_is_pad,
        )

        # Encode images for API
        content = []
        for cam_name, rgb_tensor in rgb_images.items():
            # Convert tensor to numpy
            rgb_np = rgb_tensor.cpu().numpy()
            image_b64 = encode_image(rgb_np)

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })

        # Add text prompt with camera info
        content.append({
            "type": "text",
            "text": f"Camera views: {', '.join(rgb_images.keys())}\n\n{prompt}"
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
            evaluation_text = message.content[0].text.strip()

            # Parse the response to extract score and explanation
            score, explanation = self._parse_evaluation(evaluation_text)

            return score, explanation

        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {str(e)}")
