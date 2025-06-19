import torch
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer


class ExperienceReplay(ContinualModel):
    """
    A class to handle experience replay for reinforcement learning agents.
    It stores experiences and allows sampling of random batches for training.
    """

    NAME = "experience_replay"

    def __init__(self, backbone, loss, args, transform):
        super(ExperienceReplay, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def training_process(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.

        :param inputs: batch of examples
        :param labels: ground-truth labels

        :return: the value of the loss function
        """
        real_batch_size = inputs.shape[0]

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.sample_experience(
                self.args.minibatch_size
            )
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.inference(inputs)

        self.opt.zero_grad()
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_experience(
            inputs=inputs[:real_batch_size], labels=labels[:real_batch_size]
        )

        return loss.item()
