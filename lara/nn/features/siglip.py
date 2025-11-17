import torch
import torch.nn as nn
from typing import Optional
import logging
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class SigLip(nn.Module):
    """Base class for SigLIP embedding models."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize SigLIP model and processor.

        Args:
            model_name: HuggingFace model identifier for SigLIP
            device: Device to load model on ("cuda" or "cpu")
            torch_dtype: Data type for model ("auto", "bfloat16", "float16", "float32")
            trust_remote_code: Whether to trust remote code when loading model
            output_dim: Embedding dimension. If None, will be inferred from model config
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code

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

        # Load model
        logger.info(f"Loading SigLIP model {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()

        # Load processor
        logger.info(f"Loading processor for {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Infer output_dim from model config if not provided
        if output_dim is None:
            if hasattr(self.model.config, "projection_dim"):
                self.output_dim = self.model.config.projection_dim
            elif hasattr(self.model.config, "hidden_size"):
                self.output_dim = self.model.config.hidden_size
            else:
                raise ValueError(
                    f"Could not infer output_dim from model config. "
                    f"Please provide output_dim explicitly."
                )
        else:
            self.output_dim = output_dim

        logger.info(f"Initialized SigLIP with model: {type(self.model).__name__}, output_dim: {self.output_dim}")



class SubtaskSigLip(SigLip):
    """Encode subtask text using SigLIP text encoder."""

    def forward(self, subtasks: list[str]) -> torch.Tensor:
        """
        Encode a batch of subtask texts using SigLIP text encoder.

        Args:
            subtasks: List of subtask description strings

        Returns:
            Text embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Process text inputs
        inputs = self.processor(
            text=subtasks,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get text embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # SigLIP returns text_embeds
            text_embeddings = outputs.text_embeds

        return text_embeddings


class RGBSigLip(SigLip):
    """Encode RGB images using SigLIP image encoder."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        fusion_method: str = "concat",
        output_dim: Optional[int] = None,
        num_cameras: int = 1,
    ):
        """
        Initialize RGBSigLip encoder.

        Args:
            model_name: HuggingFace model identifier for SigLIP
            device: Device to load model on
            torch_dtype: Data type for model
            trust_remote_code: Whether to trust remote code
            fusion_method: How to combine multi-view embeddings ("concat", "mean", "max")
            output_dim: Base embedding dimension. If None, will be inferred from model config
            num_cameras: Number of cameras for multi-view fusion. Used to compute final output_dim
                        when fusion_method is "concat"
        """
        super().__init__(model_name, device, torch_dtype, trust_remote_code, output_dim)
        self.fusion_method = fusion_method
        self.num_cameras = num_cameras

        # Adjust output_dim based on fusion method
        if self.fusion_method == "concat":
            # When concatenating, output_dim is multiplied by number of cameras
            self._base_output_dim = self.output_dim
            self.output_dim = self.output_dim * num_cameras
        # For "mean" and "max" fusion, output_dim stays the same

    def forward(self, rgb_images: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of multi-view RGB images using SigLIP image encoder.

        Expects images to be pre-normalized to [0, 1] range (from process_data).

        Args:
            rgb_images: Dictionary of camera name to image tensors.
                       Each tensor shape: (batch_size, time_steps, channels, height, width)
                       Already normalized to [0, 1] from process_data function.

        Returns:
            Fused image embeddings tensor of shape:
            - (batch_size, num_cameras * embedding_dim) if fusion_method='concat'
            - (batch_size, embedding_dim) if fusion_method='mean' or 'max'
        """
        if not rgb_images:
            raise ValueError("rgb_images dictionary cannot be empty")

        all_embeddings = []

        # Process each camera's images
        for cam_name, img_tensor in rgb_images.items():
            logger.debug(f"Processing camera: {cam_name}, shape: {img_tensor.shape}")

            # Extract latest image from batch
            # Expect shape: (batch_size, time_steps, channels, height, width)
            # Take the last timestep: shape becomes (batch_size, channels, height, width)
            if img_tensor.ndim == 5:  # (B, T, C, H, W)
                latest_imgs = img_tensor[:, -1, :, :, :]
            elif img_tensor.ndim == 4:  # (B, C, H, W)
                latest_imgs = img_tensor
            else:
                raise ValueError(
                    f"Expected image tensor of shape (B, T, C, H, W) or (B, C, H, W), "
                    f"got shape {img_tensor.shape}"
                )

            logger.debug(f"Image shape before processor: {latest_imgs.shape}")

            # Process through SigLIP processor and model
            inputs = self.processor(
                images=latest_imgs,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get image embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeddings = outputs.image_embeds

            logger.debug(f"Image embeddings shape from {cam_name}: {image_embeddings.shape}")
            all_embeddings.append(image_embeddings)

        # Fuse embeddings from multiple cameras
        if self.fusion_method == "concat":
            # Concatenate embeddings from all cameras along feature dimension
            fused_embeddings = torch.cat(all_embeddings, dim=1)
        elif self.fusion_method == "mean":
            # Average embeddings from all cameras
            stacked = torch.stack(all_embeddings, dim=0)
            fused_embeddings = torch.mean(stacked, dim=0)
        elif self.fusion_method == "max":
            # Max pool embeddings across cameras
            stacked = torch.stack(all_embeddings, dim=0)
            fused_embeddings = torch.max(stacked, dim=0)[0]
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        logger.debug(f"Final fused embeddings shape: {fused_embeddings.shape}")
        return fused_embeddings