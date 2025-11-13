import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class Writer:
    """Writer for saving evaluation ground truth in JSONL format with metadata."""

    def __init__(
        self,
        output_dir: str = "./eval_outputs",
        output_filename: str = "eval_gt.jsonl",
        metadata_filename: str = "metadata.json",
    ):
        """
        Initialize the Writer.

        Args:
            output_dir: Directory to save output files
            output_filename: Name of the JSONL output file
            metadata_filename: Name of the metadata JSON file
        """
        self.output_dir = Path(output_dir)
        self.output_filepath = self.output_dir / output_filename
        self.metadata_filepath = self.output_dir / metadata_filename

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "total_entries": 0,
            "tasks": {},
            "episodes": {},
        }

        # Create/truncate the output file
        self.output_filepath.touch()
        self._save_metadata()

    def write(
        self,
        task_id: Any,
        frame_id: Any,
        episode_id: Any,
        timestamp: float,
        subtask_gt: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        Write a single evaluation entry to the JSONL file.

        Args:
            task_id: Task identifier
            frame_id: Frame identifier within episode
            episode_id: Episode identifier
            timestamp: Timestamp in seconds
            subtask_gt: Ground truth subtask description
            extra_fields: Optional additional fields to include
        """
        # Build entry
        entry = {
            "task_id": str(task_id),
            "episode_id": str(episode_id),
            "frame_id": int(frame_id) if isinstance(frame_id, (int, float)) else str(frame_id),
            "timestamp": float(timestamp),
            "subtask_gt": subtask_gt,
        }

        # Add extra fields if provided
        if extra_fields:
            entry.update(extra_fields)

        # Append to JSONL file
        with open(self.output_filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Update metadata
        self._update_metadata(task_id, episode_id)

    def _update_metadata(self, task_id: Any, episode_id: Any):
        """
        Update metadata tracking.

        Args:
            task_id: Task identifier
            episode_id: Episode identifier
        """
        task_id_str = str(task_id)
        episode_id_str = str(episode_id)

        # Update total count
        self.metadata["total_entries"] += 1

        # Track tasks
        if task_id_str not in self.metadata["tasks"]:
            self.metadata["tasks"][task_id_str] = 0
        self.metadata["tasks"][task_id_str] += 1

        # Track episodes
        if episode_id_str not in self.metadata["episodes"]:
            self.metadata["episodes"][episode_id_str] = {
                "task_id": task_id_str,
                "frame_count": 0,
            }
        self.metadata["episodes"][episode_id_str]["frame_count"] += 1

    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_filepath, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def finalize(self):
        """
        Finalize writing and save final metadata.
        Call this when done writing all entries.
        """
        self.metadata["completed_at"] = datetime.now().isoformat()
        self._save_metadata()

        print(f"✓ Saved {self.metadata['total_entries']} entries to {self.output_filepath}")
        print(f"✓ Metadata saved to {self.metadata_filepath}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically finalize."""
        self.finalize()