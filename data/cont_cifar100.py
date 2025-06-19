from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.resnet import resnet18
import torch.nn.functional as F
from data.utils.continual_dataset import ContinualDataset, get_masked_loaders


class MyCIFAR100(CIFAR100):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ContinualCIFAR100(ContinualDataset):
    NAME = "continual_cifar100"
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    def get_data_loaders(self, masked_loaders=True) -> Tuple[DataLoader, DataLoader]:
        tranform = self.TRANSFORM

        train_dataset = MyCIFAR100(
            root=self.args.data_path, train=True, transform=tranform, download=True
        )
        test_dataset = MyCIFAR100(
            root=self.args.data_path, train=False, transform=tranform, download=True
        )

        train_loader = None
        test_loader = None
        # If masked loaders are not requested, we load the full dataset
        # otherwise we create masked loaders
        # based on the current task and number of classes per task.
        if masked_loaders:
            if isinstance(train_dataset.targets, list):
                train_dataset.targets = torch.tensor(
                    train_dataset.targets, dtype=torch.long
                )
            if isinstance(test_dataset.targets, list):
                test_dataset.targets = torch.tensor(
                    test_dataset.targets, dtype=torch.long
                )
            train_loader, test_loader = get_masked_loaders(
                train_dataset, test_dataset, self
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=2,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=2,
            )

        return train_loader, test_loader

    @staticmethod
    def get_backbone():
        return resnet18(
            ContinualCIFAR100.N_CLASSES_PER_TASK * ContinualCIFAR100.N_TASKS
        )

    @staticmethod
    def get_loss():
        return F.cross_entropy
