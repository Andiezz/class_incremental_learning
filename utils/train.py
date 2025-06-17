from argparse import Namespace

import torch
from data.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.tools import progress_bar


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Masks output tensor by setting values for classes not in current task to -infinity.

    :param outputs: model prediction tensor
    :param dataset: continual learning dataset
    :param k: current task index
    """
    THIS_TASK_START = k * dataset.N_CLASSES_PER_TASK
    THIS_TASK_END = (k + 1) * dataset.N_CLASSES_PER_TASK

    outputs[:, :THIS_TASK_START] = -float("inf")
    outputs[:, THIS_TASK_END:] = -float("inf")


def evaluate(
    model: ContinualModel, dataset: ContinualDataset, current_task: int
) -> list[float]:
    """
    Evaluate the model on the current task.
    :param model: the continual learning model
    :param dataset: the continual dataset
    :param task: the current task index
    :return: the accuracy of the model on the current task
    """
    model.net.eval()
    model.net.to(model.device)
    test_loaders = dataset.test_loaders
    correct = 0
    total = 0

    task_wise_accuracy = [0.0] * dataset.N_TASKS
    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            # TODO
            # BUG
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model.inference(inputs)

                # Mask outputs for classes not in the current task
                mask_classes(outputs, dataset, task)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = correct / total if total > 0 else 0.0

                task_wise_accuracy[task - 1] = accuracy

    # Print task-wise accuracy
    for i, acc in enumerate(task_wise_accuracy):
        if i <= current_task:
            print(f"Task {i + 1} accuracy: {acc:.4f}")

    return task_wise_accuracy


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    Train the model on the current task.
    :param model: the continual learning model
    :param dataset: the continual dataset
    :param args: command line arguments
    """
    model.net.to(model.device)

    tasks = dataset.N_TASKS

    task_max_accuracy = [0.0] * tasks
    task_wise_accuracy = [[0.0] * tasks for _ in range(tasks)]

    print(
        f"Training on {tasks} tasks with {dataset.N_CLASSES_PER_TASK} classes per task."
    )

    for current_task in range(tasks):
        print(f"Training on task {current_task + 1}/{tasks}")
        train_loader, _ = dataset.get_data_loaders(masked_loaders=True)

        for epoch in range(args.epochs):
            model.net.train()

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                loss = model.training_process(inputs, labels)

                progress_bar(i, len(train_loader), epoch + 1, current_task + 1, loss)

        # Evaluate the model on the current task
        curr_task_wise_accuracy = evaluate(model, dataset, current_task)
        task_wise_accuracy[current_task] = curr_task_wise_accuracy[: current_task + 1]

        # Update the maximum accuracy for the current task
        task_max_accuracy[current_task] = max(
            task_max_accuracy[current_task], curr_task_wise_accuracy[current_task]
        )

    # Calculate forgetting metrics
    print("\nForgetting metrics:")
    for task_idx in range(tasks - 1):  # Skip last task as it has no forgetting
        max_acc = task_max_accuracy[task_idx]
        final_acc = task_wise_accuracy[-1][task_idx]
        forgetting = max_acc - final_acc
        print(
            f"Task {task_idx + 1}: Max accuracy: {max_acc:.4f}, Final accuracy: {final_acc:.4f}, Forgetting: {forgetting:.4f}"
        )

    # Calculate final average accuracy by averaging the accuracy of the last task
    # over all tasks it has seen so far
    print("\nFinal accuracy across all tasks:")
    final_task_accuracies = task_wise_accuracy[-1]
    tasks_seen = len(final_task_accuracies)
    final_average_accuracy = sum(final_task_accuracies) / tasks_seen
    print(f"Average accuracy: {final_average_accuracy:.4f}")
