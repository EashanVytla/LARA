import torch
import logging
import importlib
from typing import Optional
from omegaconf import DictConfig
from lara.utils import format_state_action_history

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

class Planner():
    def __init__(
        self,
        model_class: str,
        processor_class: str,
        model_name: str,
        prompts: DictConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize the planner with a vision-language model and processor.

        Args:
            model_class: Full path to model class (e.g., "transformers.Qwen2VLForConditionalGeneration")
            processor_class: Full path to processor class (e.g., "transformers.AutoProcessor")
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen2-VL-2B-Instruct")
            prompts: Prompt configuration from prompts.yaml
            device: Device to load model on ("cuda" or "cpu")
            torch_dtype: Data type for model ("auto", "bfloat16", "float32", "float16")
            trust_remote_code: Whether to trust remote code when loading model
        """
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.prompts = prompts  # Access planner prompts

        # Parse torch dtype
        if torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown torch_dtype: {torch_dtype}")

        # Load model class dynamically
        model_module_path, model_class_name = model_class.rsplit(".", 1)
        ModelClass = getattr(importlib.import_module(model_module_path), model_class_name)

        # Load processor class dynamically
        processor_module_path, processor_class_name = processor_class.rsplit(".", 1)
        ProcessorClass = getattr(importlib.import_module(processor_module_path), processor_class_name)

        logger.info(f"Loading model {model_name} with {model_class_name}...")

        # Load model
        self.model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )

        # Load processor
        logger.info(f"Loading processor for {model_name}...")
        self.processor = ProcessorClass.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        logger.info(f"Initialized Planner with model: {type(self.model).__name__}")

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
        states,
        actions: torch.Tensor,
        use_stuck_detection: bool = True,
        velocity_threshold: float = 0.01,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Build the prompt for subtask generation using configurable templates.

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
        p = self.prompts  # Shorthand for prompt templates

        logger.info(f"Task: {task}")

        # Build prompt from configurable components
        prompt = f"{p.intro}\n"
        prompt += f"{p.task_line.format(task=task)}\n\n"
        prompt += f"{p.context}\n\n"
        prompt += f"{p.goal}\n\n"

        # Add state-action history (complex formatting stays in Python)
        prompt += format_state_action_history(states, actions, state_is_pad, action_is_pad)
        prompt += "\n\n"

        # Add stuck warning if detected
        # if use_stuck_detection and self._detect_stuck(states, actions, velocity_threshold):
        # prompt += f"{p.stuck_warning}\n\n"

        # Add guidelines and output label
        prompt += f"{p.guidelines}\n\n"
        prompt += f"{p.output_label}"
        prompt += f"{p.output_format}"

        return prompt

    def predict_subtask(
        self,
        rgb_images: dict[str, torch.Tensor],
        task: str,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_stuck_detection: bool = True,
        velocity_threshold: float = 0.01,
        state_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate subtask prediction using the vision-language model.

        Args:
            rgb_images: Dictionary of RGB images {camera_name: tensor}, each shape (B, T, C, H, W) or (H, W, 3)
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
        logger.info("Building prompt")

        batch_size = next(iter(rgb_images.values())).shape[0]
        text_inputs = []
        image_lists = []

        logger.info(f"Batch size: {batch_size}")

        for i in range(batch_size):
            # Build text prompt
            text_prompt = self._build_prompt(
                task=task,
                states=states,
                actions=actions,
                use_stuck_detection=use_stuck_detection,
                velocity_threshold=velocity_threshold,
                state_is_pad=state_is_pad,
                action_is_pad=action_is_pad,
            )

            logger.info("Prompt building is complete")

            # Prepare images for Qwen2VL
            # Qwen2VL processor can accept tensors directly without PIL conversion
            images_list = []
            for cam_name, img_tensor in rgb_images.items():
                logger.info("Processing image tensor")
                # Remove batch and sequence dimensions if present
                # Expected shape: (B, T, C, H, W) or (B, C, H, W) or (T, C, H, W) or (C, H, W)
                latest_img_tensor = img_tensor[i, -1]

                logger.info(f"Image tensor shape: {latest_img_tensor.shape}")

                # Normalize to [0, 1] range if needed (processor handles normalization)
                if latest_img_tensor.dtype == torch.float32 or latest_img_tensor.dtype == torch.float16:
                    logger.info("Normalizing image to [0, 1] range")
                    if latest_img_tensor.max() > 1.0:
                        latest_img_tensor = latest_img_tensor / 255.0

                logger.info("Image tensor processing complete")
                images_list.append(latest_img_tensor)

            # Create Qwen2VL message format with images
            # Qwen2VL uses a chat format with image tokens
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images_list],
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            # Process inputs through the processor
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            text_inputs.append(text_input)
            image_lists.append(images_list)

        logger.info(f"Image Lists List Size: {len(image_lists)}")

        # Prepare inputs for the model
        inputs = self.processor(
            text=text_inputs,
            images=image_lists,
            return_tensors="pt",
            padding=True,
        )

        logger.info(type(inputs))

        # Move inputs to the same device as the model
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Max tokens for subtask description
                do_sample=False,  # Use greedy decoding for consistency
            )

        logger.info(generated_ids.shape)

        # Decode the generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        subtask_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        logger.info(len(subtask_text))

        return subtask_text
