import argparse
from data import NAMES as DATASET_NAMES
from models import get_all_models


def get_args() -> None:
    """
    Adds the arguments used by all the models.

    :param parser: the parser instance
    """
    parser = argparse.ArgumentParser(description="Continual Learning Training")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_NAMES,
        default="continual_cifar100",
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="elastic_weight_consolidation",
        help="Model name.",
        choices=get_all_models(),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs for each task.",
    )
    parser.add_argument(
        "--checkpoint-path",
        "-c",
        type=str,
        default="checkpoints",
        help="checkpoint folder",
    )
    parser.add_argument(
        "--resume-training",
        "-r",
        type=bool,
        default=False,
        help="Continue training or not",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--momentum", "-m", type=float, default=0.9, help="momentum of optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=5e-4,
        help="weight decay of optimizer",
    )
    parser.add_argument(
        "--data_path", "-dp", type=str, default="datasets", help="Path to the dataset."
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="Size of the experience replay buffer.",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=8,
        help="Size of the mini-batch buffer for training.",
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=100,
        help="Regularization strength for Elastic Weight Consolidation.",
    )
    parser.add_argument("--log-path", "-l", type=str, default="logs", help="log folder")
    parser.add_argument("--seed", type=int, default=612, help="The random seed.")

    args = parser.parse_args()
    return args
