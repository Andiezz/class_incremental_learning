from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple
from argparse import Namespace
from torchvision import datasets


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

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Generate and return training and test data loaders for the current task.

        Stores the current training loader and all test loaders in self.

        Returns:
            tuple: A tuple containing (train_loader, test_loader) for the current task.
        """
        pass


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
    n_classes_per_task = setting.N_CLASSES_PER_TASK

    # get the mask for the current task
    train_mask = (train_dataset.targets % 1000 >= current_task * n_classes_per_task) & (
        train_dataset.targets % 1000 < (current_task + 1) * n_classes_per_task
    )
    test_mask = (test_dataset.targets % 1000 >= current_task * n_classes_per_task) & (
        test_dataset.targets % 1000 < (current_task + 1) * n_classes_per_task
    )

    # apply the mask to the datasets
    train_dataset.targets = train_dataset.targets[train_mask]
    train_dataset.data = train_dataset.data[train_mask]

    test_dataset.targets = test_dataset.targets[test_mask]
    test_dataset.data = test_dataset.data[test_mask]

    # create data loaders
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

    # store the current train loader and all test loaders
    setting.test_loaders.append(test_loader)

    setting.current_task += 1
    return train_loader, test_loader
