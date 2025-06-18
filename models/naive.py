import torch
from models.utils.continual_model import ContinualModel


class Naive(ContinualModel):
    """Naive method for continual learning.

    This method does not implement any continual learning strategy.
    It simply trains the model on the current task without considering previous tasks.
    """
    NAME = "naive"
    
    def __init__(self, backbone, loss, args, transform):
        super(Naive, self).__init__(backbone, loss, args, transform)

    def training_process(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.

        :param inputs: batch of examples
        :param labels: ground-truth labels

        :return: the value of the loss function
        """
        self.net.train()
        outputs = self.inference(inputs)

        self.opt.zero_grad()
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
