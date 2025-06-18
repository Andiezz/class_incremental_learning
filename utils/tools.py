import sys
import random
import torch
import numpy as np
from typing import Union
from datetime import datetime


def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int = 612) -> None:
    """
    Sets the seeds at a certain value.

    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def progress_bar(
    i: int, max_iter: int, epoch: Union[int, str], task_number: int, loss: float
) -> None:
    """
    Prints out the progress bar on the stderr file.

    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    if not (i + 1) % 10 or (i + 1) == max_iter:
        progress = min(float((i + 1) / max_iter), 1)
        progress_bar = ("█" * int(50 * progress)) + ("┈" * (50 - int(50 * progress)))
        print(
            "\r[ {} ] Task {} | epoch {}: |{}| loss: {}".format(
                datetime.now().strftime("%m-%d | %H:%M"),
                task_number,
                epoch,
                progress_bar,
                round(loss / (i + 1), 8),
            ),
            file=sys.stderr,
            end="",
            flush=True,
        )
        if (i + 1) == max_iter:
            print(file=sys.stderr, end="\n", flush=True)
