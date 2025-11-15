import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dotenv import load_dotenv
import torch
from tqdm import tqdm
from pathlib import Path
import traceback
import time
import numpy as np

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def setup(cfg: DictConfig):
    """
    Use hydra instantiate to instantiate the data_module, planner, evaluator, writer.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (data_module, planner, evaluator, writer)
    """
    # Load environment variables
    load_dotenv()

    # Instantiate data module
    print("Initializing data module...")
    data_module = instantiate(cfg.data)
    data_module.setup(stage="fit")  # Setup with validation split

    # Instantiate planner
    print("Initializing planner...")
    planner = instantiate(cfg.planner)

    # Instantiate evaluator
    print("Initializing evaluator...")
    evaluator = instantiate(cfg.evaluator)

    # Instantiate writer
    print("Initializing writer...")
    writer = instantiate(cfg.writer)

    return data_module, planner, evaluator, writer


def compute_metrics(scores: list[float]) -> dict:
    """
    Compute evaluation metrics from collected scores.

    Args:
        scores: List of scores from evaluations

    Returns:
        Dictionary containing computed metrics
    """
    if not scores:
        return {
            "num_samples": 0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "median_score": 0.0,
        }

    scores_array = np.array(scores)
    return {
        "num_samples": len(scores),
        "mean_score": float(np.mean(scores_array)),
        "std_score": float(np.std(scores_array)),
        "min_score": float(np.min(scores_array)),
        "max_score": float(np.max(scores_array)),
        "median_score": float(np.median(scores_array)),
        "percentile_25": float(np.percentile(scores_array, 25)),
        "percentile_75": float(np.percentile(scores_array, 75)),
    }


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
    # BehaviorLeRobotDataset uses 'index' instead of 'frame_index'
    frame_index = batch["index"][0].item()
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


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="eval_gt_config",
)
def main(cfg: DictConfig):
    """
    Use planner to predict subtasks, evaluator to evaluate them, and writer to save results.
    Iterate through the validation set, generate predictions, evaluate them, and write to file.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Subtask Prediction & Evaluation")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Setup components
    data_module, planner, evaluator, writer = setup(cfg)

    # Get validation dataloader
    val_dataloader = data_module.val_dataloader()
    total_batches = len(val_dataloader)
    max_iterations = cfg.eval.get("max_iterations", None)

    if max_iterations is not None:
        batches_to_process = min(max_iterations, total_batches)
        print(f"\nProcessing {batches_to_process} batches (max_iterations={max_iterations}) from validation set...")
    else:
        batches_to_process = total_batches
        print(f"\nProcessing all {total_batches} batches from validation set...")

    # Track scores for metrics computation
    all_scores = []

    # Process each batch
    try:
        with writer:  # Use context manager for automatic finalization
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Predicting & Evaluating", total=batches_to_process)):
                # Check if we've reached max_iterations
                if max_iterations is not None and batch_idx >= max_iterations:
                    print(f"\nReached max_iterations ({max_iterations}). Stopping evaluation.")
                    break
                try:
                    # Extract data from batch
                    data = extract_batch_data(batch, cameras=cfg.data.cameras)

                    # Step 1: Predict subtask using Planner (Qwen2VL)
                    print(f"\n[Batch {batch_idx}] Predicting subtask with Planner...")
                    predicted_subtask = planner.predict_subtask(
                        rgb_images=data["rgb_images"],  # Planner expects tensors
                        task=data["task"],
                        states=data["states"],
                        actions=data["actions"],
                        use_stuck_detection=cfg.eval.use_stuck_detection,
                        velocity_threshold=cfg.eval.velocity_threshold,
                        state_is_pad=data["state_is_pad"],
                        action_is_pad=data["action_is_pad"],
                    )
                    print(f"Predicted subtask: {predicted_subtask}")

                    # Step 2: Evaluate the prediction using Evaluator (Claude)
                    print(f"[Batch {batch_idx}] Evaluating prediction with Claude...")
                    score, explanation = evaluator.evaluate_prediction(
                        subtask=predicted_subtask,
                        rgb_images=data["rgb_images"],  # Evaluator expects tensors
                        task=data["task"],
                        states=data["states"],
                        actions=data["actions"],
                        use_stuck_detection=cfg.eval.use_stuck_detection,
                        velocity_threshold=cfg.eval.velocity_threshold,
                        state_is_pad=data["state_is_pad"],
                        action_is_pad=data["action_is_pad"],
                    )
                    print(f"Evaluation - Score: {score}, Explanation: {explanation}")

                    # Collect score for metrics
                    all_scores.append(score)

                    # Add delay to avoid rate limiting (adjust as needed)
                    time.sleep(1.0)  # 1 second delay between API calls

                    # Write prediction and evaluation to file
                    writer.write(
                        task_id=data["task_index"],
                        frame_id=data["frame_index"],
                        episode_id=data["episode_index"],
                        timestamp=data["timestamp"],
                        subtask_pred=predicted_subtask,
                        score=score,
                        explanation=explanation,
                        extra_fields={
                            "task_name": data["task"],
                        },
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
                    print(f"\nError processing batch {batch_idx}:")
                    traceback.print_exc()
                    continue

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Finalizing outputs...")
        writer.finalize()

    # Compute and save metrics
    print("\n" + "=" * 60)
    print("Computing evaluation metrics...")
    print("=" * 60)

    metrics = compute_metrics(all_scores)

    # Print metrics to console
    print("\nEvaluation Metrics:")
    print(f"  Number of samples: {metrics['num_samples']}")
    print(f"  Mean score: {metrics['mean_score']:.4f}")
    print(f"  Std score: {metrics['std_score']:.4f}")
    print(f"  Min score: {metrics['min_score']:.4f}")
    print(f"  Max score: {metrics['max_score']:.4f}")
    print(f"  Median score: {metrics['median_score']:.4f}")
    print(f"  25th percentile: {metrics['percentile_25']:.4f}")
    print(f"  75th percentile: {metrics['percentile_75']:.4f}")

    # Save metrics to JSON file
    metrics_path = Path(cfg.writer.output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("✓ Subtask prediction and evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()