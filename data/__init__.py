from data.cont_cifar100 import ContinualCIFAR100
from data.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    ContinualCIFAR100.NAME: ContinualCIFAR100,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.

    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
