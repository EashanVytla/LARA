import json
import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class TaskPrompt:
    """
    Lookup table for mapping episode indices to task prompts.

    Loads task prompts from episodes.jsonl where each episode contains a list of task descriptions.
    Provides efficient O(1) lookup of task prompts by episode index.
    """

    def __init__(self, data_path: str):
        """
        Initialize TaskPrompt loader.

        Args:
            data_path: Path to the data directory containing meta/episodes.jsonl
        """
        self._data_path = os.path.expanduser(data_path)
        self._data_path = os.path.join(self._data_path, "2025-challenge-demos")
        self._episode_to_prompts: Dict[int, List[str]] = {}
        self._load_episode_prompts()

    def _load_episode_prompts(self) -> None:
        """Load task prompts from episodes.jsonl file."""
        episodes_file = os.path.join(self._data_path, "meta", "episodes.jsonl")

        if not os.path.exists(episodes_file):
            logger.warning(f"Episodes file not found: {episodes_file}")
            return

        try:
            with open(episodes_file, "r") as f:
                for line in f:
                    if line.strip():
                        episode_data = json.loads(line)
                        episode_index = episode_data.get("episode_index")
                        tasks = episode_data.get("tasks", [])

                        if episode_index is not None:
                            self._episode_to_prompts[episode_index] = tasks

            logger.info(
                f"Loaded task prompts for {len(self._episode_to_prompts)} episodes "
                f"from {episodes_file}"
            )
        except Exception as e:
            logger.error(f"Error loading episodes.jsonl: {e}")

    def get_prompts(self, episode_index: int) -> List[str]:
        """
        Get task prompts for a given episode.

        Args:
            episode_index: The episode index

        Returns:
            List of task prompt strings for the episode. Returns empty list if not found.
        """
        return self._episode_to_prompts.get(episode_index, [])

    def get_prompt(self, episode_index: int, task_index: int = 0) -> str:
        """
        Get a single task prompt for a given episode.

        Args:
            episode_index: The episode index
            task_index: The task index within the episode (default 0, first task)

        Returns:
            Task prompt string. Returns empty string if not found.
        """
        prompts = self.get_prompts(episode_index)
        if task_index < len(prompts):
            return prompts[task_index]
        return ""

    def get_prompts_padded(
        self, episode_index: int, max_length: int, pad_token: str = ""
    ) -> List[str]:
        """
        Get task prompts for a given episode, padded to max_length.

        Args:
            episode_index: The episode index
            max_length: Maximum length to pad to
            pad_token: Padding string (default empty string)

        Returns:
            List of task prompts padded to max_length with pad_token
        """
        prompts = self.get_prompts(episode_index)
        if len(prompts) < max_length:
            prompts = prompts + [pad_token] * (max_length - len(prompts))
        return prompts[:max_length]
