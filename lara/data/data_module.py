import importlib
import os
import logging
from lara.data.dataset import DummyDataset
from lara.data.task_prompt import TaskPrompt
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional

logger = logging.getLogger(__name__)


class BehaviorDataModule(LightningDataModule):
    def __init__(
        self,
        *args,
        data_path: str,
        task_name: list[str],
        batch_size: int,
        val_batch_size: Optional[int],
        val_split_ratio: float,
        dataloader_num_workers: int,
        seed: int,
        max_num_demos: Optional[int] = None,
        dataset_class: str,
        split_dataset: bool = True,  # If False, use all data for val only
        **kwargs,
    ):
        super().__init__()
        self._data_path = os.path.expanduser(data_path)
        self._task_name = task_name
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        self._max_num_demos = max_num_demos
        self._seed = seed
        self._dataset_class = dataset_class
        self._split_dataset = split_dataset
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

        # Initialize task prompt lookup
        self._task_prompt_loader = TaskPrompt(self._data_path)

    def _collate_with_task_prompts(self, batch_list):
        """
        Custom collate function that adds task prompts to the batch.

        Looks up task prompts from episodes.jsonl using episode_index and adds them to the batch.
        Task prompts are padded to match batch size.
        """
        from torch.utils.data.dataloader import default_collate

        # Standard collation
        batch = default_collate(batch_list)

        # Extract episode indices from batch (should be a tensor)
        episode_indices = batch.get("episode_index")

        if episode_indices is not None:
            # Convert to list of integers if it's a tensor
            if hasattr(episode_indices, "tolist"):
                episode_indices = episode_indices.tolist()
            elif not isinstance(episode_indices, list):
                episode_indices = [episode_indices]

            # Get task prompts for each episode in the batch
            task_prompts = []
            for episode_idx in episode_indices:
                prompts = self._task_prompt_loader.get_prompts(int(episode_idx))
                # Take first prompt if multiple tasks, else empty string
                prompt = prompts[0] if prompts else ""
                task_prompts.append(prompt)

            batch["task_prompt"] = task_prompts

        return batch

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # get dataset class module
            module_path, class_name = self._dataset_class.rsplit(".", 1)
            DatasetClassModule = getattr(importlib.import_module(module_path), class_name)

            # Check if this is BehaviorLeRobotDataset
            is_lerobot = "LeRobot" in class_name or "lerobot" in self._dataset_class.lower()

            if is_lerobot:
                # BehaviorLeRobotDataset: split episodes into train/val
                # Get episodes from kwargs (None means all episodes)
                episodes = self._kwargs.get('episodes', None)

                # Handle episodes parameter: int -> range(N), list -> list, None -> None
                if episodes is not None:
                    if isinstance(episodes, int):
                        # If int, treat as range(N) -> [0, 1, 2, ..., N-1]
                        episodes = list(range(episodes))
                    elif isinstance(episodes, (list, tuple)):
                        # If already a list/tuple, keep as is
                        episodes = list(episodes)
                    else:
                        # Handle any other iterable (OmegaConf ListConfig, etc.)
                        episodes = list(episodes)

                if not self._split_dataset:
                    # No split: use all data for validation only
                    train_episodes = None  # No training data
                    val_episodes = episodes
                elif episodes is not None:

                    # Split episodes into train/val
                    train_episodes, val_episodes = train_test_split(
                        episodes,
                        test_size=self._val_split_ratio,
                        random_state=self._seed,
                    )
                else:
                    # episodes=None means use all data
                    # For train/val split, we'll use the same dataset for both
                    # (the actual split would need to be done differently)
                    train_episodes = None
                    val_episodes = None

                # Filter out kwargs that are specific to BehaviorIterableDataset
                lerobot_incompatible_keys = {'downsample_factor', 'robot_type', 'obs_window_size',
                                             'ctx_len', 'use_task_info', 'task_info_range',
                                             'multi_view_cameras'}

                # Create train dataset (only if splitting)
                if self._split_dataset and train_episodes is not None:
                    train_kwargs = {k: v for k, v in self._kwargs.items()
                                   if k not in lerobot_incompatible_keys}
                    train_kwargs['episodes'] = train_episodes
                    logger.info(f"Creating train dataset with episodes: {train_episodes}")
                    self._train_dataset = DatasetClassModule(
                        root=self._data_path,
                        **train_kwargs,
                    )
                else:
                    # Create a dummy train dataset to satisfy assertions
                    self._train_dataset = None

                # Create val dataset
                val_kwargs = {k: v for k, v in self._kwargs.items()
                             if k not in lerobot_incompatible_keys}
                val_kwargs['episodes'] = val_episodes
                logger.info(f"Creating val dataset with episodes: {val_episodes}")
                self._val_dataset = DatasetClassModule(
                    root=self._data_path,
                    **val_kwargs,
                )
            else:
                # BehaviorIterableDataset: uses traditional demo_keys approach
                all_demo_keys = DatasetClassModule.get_all_demo_keys(self._data_path, self._task_name)
                # limit number of demos
                if self._max_num_demos is not None:
                    all_demo_keys = all_demo_keys[: self._max_num_demos]
                self._train_demo_keys, self._val_demo_keys = train_test_split(
                    all_demo_keys,
                    test_size=self._val_split_ratio,
                    random_state=self._seed,
                )
                # initialize datasets
                self._train_dataset = DatasetClassModule(
                    *self._args,
                    **self._kwargs,
                    data_path=self._data_path,
                    demo_keys=self._train_demo_keys,
                    seed=self._seed,
                )
                self._val_dataset = DatasetClassModule(
                    *self._args,
                    **self._kwargs,
                    data_path=self._data_path,
                    demo_keys=self._val_demo_keys,
                    seed=self._seed,
                )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self._collate_with_task_prompts,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=min(self._val_batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self._collate_with_task_prompts,
        )

    def test_dataloader(self) -> DataLoader:
        """
        For test_step(), simply returns a dummy dataset.
        """
        return DataLoader(DummyDataset())

    def on_train_epoch_start(self) -> None:
        # set epoch for train dataset, which will trigger shuffling
        assert self._train_dataset is not None and self.trainer is not None
        self._train_dataset.epoch = self.trainer.current_epoch