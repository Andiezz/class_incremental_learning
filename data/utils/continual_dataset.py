from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple, List
from argparse import Namespace
from torchvision import datasets
import torch
import numpy as np


class ContinualDataset:
    """
    Continual dataset base class.
    """

    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loader = None
        self.test_loaders = []
        self.current_task = 0
        self.args = args
        self.class_order = None  # Store the random class order
        self.task_classes = None  # Store which classes belong to each task

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Generate and return training and test data loaders for the current task.

        Stores the current training loader and all test loaders in self.

        Returns:
            tuple: A tuple containing (train_loader, test_loader) for the current task.
        """
        pass

    def initialize_class_order(self, dataset_targets) -> None:
        """
        Initialize random class order for continual learning tasks.

        :param dataset_targets: Target labels from the dataset
        """
        if self.class_order is None:
            unique_classes = torch.unique(torch.tensor(dataset_targets), dim=0).tolist()
            unique_classes.sort()

            np.random.seed(self.args.seed)

            # Create a copy and shuffle randomly for generalization
            self.class_order = unique_classes.copy()
            np.random.shuffle(self.class_order)

            # Split classes into tasks to ensure non-overlapping classes
            self.task_classes = []
            for task_id in range(self.N_TASKS):
                start_idx = task_id * self.N_CLASSES_PER_TASK
                end_idx = (task_id + 1) * self.N_CLASSES_PER_TASK
                task_classes = self.class_order[start_idx:end_idx]
                self.task_classes.append(task_classes)

            print(f"Original classes: {unique_classes}")
            print(f"Shuffled class order: {self.class_order}")
            print(f"Task classes: {self.task_classes}")

    def get_task_classes(self, task_id: int) -> List[int]:
        """
        Get the classes for a specific task.
        """
        return self.task_classes[task_id]


def get_masked_loaders(
    train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.

    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting

    :return: train and test loaders
    """
    current_task = setting.current_task

    # Initialize class order
    if setting.class_order is None:
        setting.initialize_class_order(train_dataset.targets)

    # Get classes for current task
    current_task_classes = setting.get_task_classes(current_task)
    print(f"Task {current_task} classes: {current_task_classes}")

    train_targets = train_dataset.targets.clone().detach()
    test_targets = test_dataset.targets.clone().detach()

    # Create boolean masks for current task classes
    train_mask = torch.isin(train_targets, torch.tensor(current_task_classes))
    test_mask = torch.isin(test_targets, torch.tensor(current_task_classes))

    # Apply masks to datasets
    train_dataset.targets = train_targets[train_mask].tolist()
    train_dataset.data = train_dataset.data[train_mask]

    test_dataset.targets = test_targets[test_mask].tolist()
    test_dataset.data = test_dataset.data[test_mask]

    print(
        f"Task {current_task}: {len(train_dataset.targets)} train samples, {len(test_dataset.targets)} test samples"
    )
    print(
        f"Train dataset unique classes: {torch.unique(torch.tensor(train_dataset.targets), dim=0)}"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=setting.args.batch_size,
        shuffle=True,
        num_workers=setting.args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=setting.args.batch_size,
        shuffle=False,
        num_workers=setting.args.num_workers,
    )

    # Store all test loader in array subsequently to ensure evaluating the right classed in each task
    setting.test_loaders.append(test_loader)
    setting.current_task += 1

    return train_loader, test_loader
