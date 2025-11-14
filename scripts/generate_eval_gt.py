import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dotenv import load_dotenv
import torch
from tqdm import tqdm
from pathlib import Path

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def setup(cfg: DictConfig):
    """
    Use hydra instantiate to instantiate the data_module, LLM interface, writer.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (data_module, llm_interface, writer)
    """
    # Load environment variables
    load_dotenv()

    # Instantiate data module
    print("Initializing data module...")
    data_module = instantiate(cfg.data)
    data_module.setup(stage="fit")  # Setup with validation split

    # Instantiate LLM interface
    print("Initializing LLM interface...")
    llm_interface = instantiate(cfg.llm_interface)

    # Instantiate writer
    print("Initializing writer...")
    writer = instantiate(cfg.writer)

    return data_module, llm_interface, writer


def extract_batch_data(batch: dict, cameras: list[str]):
    """
    Extract relevant data from batch.

    Args:
        batch: Batch dictionary from dataloader
        cameras: List of camera names to extract

    Returns:
        Dictionary with extracted data
    """
    # Get RGB images from all cameras
    rgb_images = {}
    for camera in cameras:
        rgb_key = f"observation.images.rgb.{camera}"
        rgb_images[camera] = batch[rgb_key][0]  # Get first item in batch (batch_size=1)

    # Get metadata
    task = batch["task"][0] if isinstance(batch["task"], list) else batch["task"]
    task_index = batch["task_index"][0].item()
    episode_index = batch["episode_index"][0].item()
    frame_index = batch["frame_index"][0].item()
    timestamp = batch["timestamp"][0].item()

    # Get state and action history
    states = batch["observation.state"][0]  # Shape: (T, state_dim)
    actions = batch["action"][0]  # Shape: (T, action_dim)

    # Get padding masks if they exist
    state_is_pad = batch.get("observation.state_is_pad", [None])[0]
    action_is_pad = batch.get("action_is_pad", [None])[0]

    return {
        "rgb_images": rgb_images,
        "task": task,
        "task_index": task_index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "states": states,
        "actions": actions,
        "state_is_pad": state_is_pad,
        "action_is_pad": action_is_pad,
    }


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="eval_gt_config")
def main(cfg: DictConfig):
    """
    Use the LLM interface, writer, and data_module.
    Iterate through the validation set, generate subtask ground truth, and write to file.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Evaluation Ground Truth Generation")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Setup components
    data_module, llm_interface, writer = setup(cfg)

    # Get validation dataloader
    val_dataloader = data_module.val_dataloader()
    print(f"\nProcessing {len(val_dataloader)} batches from validation set...")

    # Process each batch
    try:
        with writer:  # Use context manager for automatic finalization
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Generating GT")):
                try:
                    # Extract data from batch
                    data = extract_batch_data(batch, cameras=cfg.data.cameras)

                    # Convert RGB images to numpy for LLM interface
                    rgb_images_np = {cam: img.cpu().numpy() for cam, img in data["rgb_images"].items()}

                    # Generate subtask using LLM
                    subtask_gt = llm_interface.predict_subtask(
                        rgb_images=rgb_images_np,
                        task=data["task"],
                        states=data["states"],
                        actions=data["actions"],
                        use_stuck_detection=cfg.eval.use_stuck_detection,
                        velocity_threshold=cfg.eval.velocity_threshold,
                        state_is_pad=data["state_is_pad"],
                        action_is_pad=data["action_is_pad"],
                    )

                    # Write to file
                    writer.write(
                        task_id=data["task_index"],
                        frame_id=data["frame_index"],
                        episode_id=data["episode_index"],
                        timestamp=data["timestamp"],
                        subtask_gt=subtask_gt,
                        extra_fields={"task_name": data["task"]},
                    )

                    # Optional: Save images
                    if cfg.eval.save_images:
                        image_dir = Path(cfg.eval.image_dir)
                        image_dir.mkdir(parents=True, exist_ok=True)
                        # Save all camera views
                        for cam_name, rgb_img in data["rgb_images"].items():
                            image_path = (
                                image_dir
                                / f"ep{data['episode_index']}_frame{data['frame_index']}_{cam_name}.png"
                            )
                            import torchvision
                            torchvision.utils.save_image(rgb_img, image_path)

                except Exception as e:
                    print(f"\n⚠️  Error processing batch {batch_idx}: {str(e)}")
                    continue

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Finalizing outputs...")
        writer.finalize()

    print("\n" + "=" * 60)
    print("✓ Ground truth generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()