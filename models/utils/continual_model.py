import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.tools import get_device
from abc import abstractmethod


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """

    NAME = None

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        args: Namespace,
        transform: torchvision.transforms,
    ) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(
            self.net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.device = get_device()

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.

        :param x: batch of inputs
        :param task_label: some models require the task label

        :return: the result of the computation
        """
        return self.net(x)

    def training_process(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.

        :param inputs: batch of examples
        :param labels: ground-truth labels

        :return: the value of the loss function
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass
