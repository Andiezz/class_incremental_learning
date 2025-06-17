from data import get_dataset
from models import get_model
from utils.args import get_args
from utils.tools import set_random_seed
from utils.train import train


def main(args):
    print("Arguments:", args)

    if args.seed is not None:
        set_random_seed(args.seed)

    dataset = get_dataset(args)

    # continual learning
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, transform=None)

    print("BEGIN CONTINUAL TRAINING")
    train(model, dataset, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
