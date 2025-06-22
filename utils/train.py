import os
from argparse import Namespace
import torch
from data.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from torch.utils.tensorboard import SummaryWriter
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
) -> tuple[list[float], list[float]]:
    """
    Evaluate the model on the current task.
    :param model: the continual learning model
    :param dataset: the continual dataset
    :param current_task: the current task index
    :return: tuple of (task_wise_accuracy, task_wise_loss)
    """
    model.net.eval()
    test_loaders = dataset.test_loaders

    task_wise_accuracy = [0.0] * dataset.N_TASKS
    task_wise_loss = [0.0] * dataset.N_TASKS

    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            correct, total = 0, 0
            running_loss = 0.0
            samples_count = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model.inference(inputs)

                # Compute loss
                loss = model.loss(outputs, labels).item()
                running_loss += loss * inputs.size(0)
                samples_count += inputs.size(0)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate accuracy and loss for the current task
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = running_loss / samples_count if samples_count > 0 else 0.0

            task_wise_accuracy[task] = accuracy
            task_wise_loss[task] = avg_loss

    # Print task-wise metrics
    for i, (acc, loss) in enumerate(zip(task_wise_accuracy, task_wise_loss)):
        if i <= current_task:
            print(f"Task {i + 1} - accuracy: {acc:.4f}, loss: {loss:.4f}")

    return task_wise_accuracy, task_wise_loss


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    Train the model on the current task.
    :param model: the continual learning model
    :param dataset: the continual dataset
    :param args: command line arguments
    """
    model.net.to(model.device)

    # Create hierarchical directory structure for better TensorBoard organization
    # Create base directory for the model type
    model_log_dir = os.path.join(args.log_path, model.NAME)
    os.makedirs(model_log_dir, exist_ok=True)
    
    # Create params-specific identifier (without timestamp)
    if args.model == "elastic_weight_consolidation":
        params_id = f"lambda{model.ewc_lambda}_lr{args.lr}"
    elif args.model == "experience_replay":
        params_id = f"buffer{args.buffer_size}_lr{args.lr}"
    else:
        params_id = f"lr{args.lr}"
        
    # Add common parameters to all models
    params_id += f"_bs{args.batch_size}_e{args.epochs}"
    
    # Add a timestamp to ensure uniqueness
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Final run directory: model_name/params_id/timestamp/
    run_dir = os.path.join(model_log_dir, params_id, timestamp)
    writer = SummaryWriter(log_dir=run_dir)
    
    # Save simple identifier (without nested structure) for checkpoint naming
    run_id = f"{model.NAME}_{params_id}_{timestamp}"
    
    # Log all hyperparameters
    hparams = {
        # Common hyperparameters
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "buffer_size": args.buffer_size,
    }

    # Add model-specific hyperparameters
    if args.model == "elastic_weight_consolidation":
        hparams["ewc_lambda"] = model.ewc_lambda
        writer.add_scalar("Hyperparameters/ewc_lambda", model.ewc_lambda, 0)

    # Log individual hyperparameters for time-series visualization
    for name, value in hparams.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Hyperparameters/{name}", value, 0)

    tasks = dataset.N_TASKS

    # Initialize training state and metrics
    (
        task_max_accuracy,
        task_wise_accuracy,
        task_wise_eval_loss,
        start_task,
        start_epoch,
    ) = initialize_training(model, dataset, args)

    # Calculate global epoch for logging
    global_epoch = start_task * args.epochs + start_epoch
    # Track global iteration across all tasks
    global_iter = 0

    for current_task in range(start_task, tasks):
        print(f"Training on task {current_task + 1}/{tasks}")
        train_loader, _ = dataset.get_data_loaders(masked_loaders=True)

        for epoch in range(start_epoch, args.epochs):
            model.net.train()
            running_loss = 0.0
            samples_count = 0
            epoch_loss_values = []  # Store loss values for this epoch

            for iter, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                loss = model.training_process(inputs, labels)

                # Log loss value after each iteration
                writer.add_scalar(
                    f"Loss/Iteration/Task_{current_task+1}", loss, global_iter
                )
                writer.add_scalar("Loss/Iteration/Global", loss, global_iter)
                global_iter += 1

                # Accumulate loss for epoch-level logging
                running_loss += loss * inputs.size(0)
                samples_count += inputs.size(0)
                epoch_loss_values.append(loss)

                progress_bar(iter, len(train_loader), epoch + 1, current_task + 1, loss)

            # Calculate average loss for the epoch
            avg_loss = running_loss / samples_count if samples_count > 0 else 0.0

            # Log the training loss for the epoch
            writer.add_scalar(
                f"Loss/Train/Epoch/Task_{current_task+1}", avg_loss, epoch
            )
            writer.add_scalar("Loss/Train/Epoch/Global", avg_loss, global_epoch)

            # Log statistics about loss distribution in this epoch
            if epoch_loss_values:
                min_loss = min(epoch_loss_values)
                max_loss = max(epoch_loss_values)
                median_loss = sorted(epoch_loss_values)[len(epoch_loss_values) // 2]

                writer.add_scalar(
                    f"Loss/Train/Stats/Min/Task_{current_task+1}", min_loss, epoch
                )
                writer.add_scalar(
                    f"Loss/Train/Stats/Max/Task_{current_task+1}", max_loss, epoch
                )
                writer.add_scalar(
                    f"Loss/Train/Stats/Median/Task_{current_task+1}", median_loss, epoch
                )

            global_epoch += 1

            # save the model state after each epoch
            model_state = {
                "model_state_dict": model.net.state_dict(),
                "optimizer_state_dict": model.opt.state_dict(),
                "current_task": current_task,
                "epoch": epoch,
                "task_wise_accuracy": task_wise_accuracy,
                "task_max_accuracy": task_max_accuracy,
                "task_wise_eval_loss": task_wise_eval_loss,
            }
            checkpoint = os.path.join(args.checkpoint_path, run_id + "_last.pt")
            torch.save(model_state, checkpoint)

        # Evaluate the model on the current task
        curr_task_wise_accuracy, curr_task_wise_loss = evaluate(
            model, dataset, current_task
        )
        task_wise_accuracy[current_task] = curr_task_wise_accuracy[: current_task + 1]
        task_wise_eval_loss[current_task] = curr_task_wise_loss[: current_task + 1]

        # Update the maximum accuracy for the current task
        task_max_accuracy[current_task] = max(
            task_max_accuracy[current_task], curr_task_wise_accuracy[current_task]
        )

        # Log validation accuracy and loss after completing each task
        for i in range(current_task + 1):
            writer.add_scalar(
                f"Accuracy/Task_{i+1}", curr_task_wise_accuracy[i], current_task
            )
            writer.add_scalar(
                f"Loss/Validation/Task_{i+1}", curr_task_wise_loss[i], current_task
            )

        # Log average metrics across all seen tasks
        avg_accuracy = sum(curr_task_wise_accuracy[: current_task + 1]) / (
            current_task + 1
        )
        avg_loss = sum(curr_task_wise_loss[: current_task + 1]) / (current_task + 1)
        writer.add_scalar("Accuracy/Average", avg_accuracy, current_task)
        writer.add_scalar("Loss/Validation/Average", avg_loss, current_task)

        # Reset start_epoch for next task
        start_epoch = 0

        # After task
        if args.model == "elastic_weight_consolidation":
            model.after_task(train_loader)

    # Calculate and log final metrics - pass hparams here
    calculate_and_log_final_metrics(
        task_wise_accuracy, task_wise_eval_loss, task_max_accuracy, writer, hparams
    )

    # Close the SummaryWriter
    writer.close()


def calculate_and_log_final_metrics(
    task_wise_accuracy: list[list[float]],
    task_wise_eval_loss: list[list[float]],
    task_max_accuracy: list[float],
    writer: SummaryWriter,
    hparams: dict,  # Add hparams parameter
) -> tuple[float, float]:
    """
    Calculate forgetting metrics and final accuracy/loss across all tasks.

    Args:
        task_wise_accuracy: Accuracy values for each task per task
        task_wise_eval_loss: Evaluation loss values for each task per task
        task_max_accuracy: Maximum accuracy achieved for each task
        writer: TensorBoard SummaryWriter instance
        hparams: Hyperparameters dictionary

    Returns:
        tuple: (final_average_accuracy, final_average_loss)
    """
    tasks = len(task_wise_accuracy[-1])

    # Calculate forgetting metrics
    print("\nForgetting metrics:")
    for task_idx in range(tasks):
        max_acc = task_max_accuracy[task_idx]
        final_acc = task_wise_accuracy[-1][task_idx]
        forgetting = max_acc - final_acc
        print(
            f"Task {task_idx + 1}: Max accuracy: {max_acc:.4f}, Final accuracy: {final_acc:.4f}, Forgetting: {forgetting:.4f}"
        )

        # Log forgetting metrics
        writer.add_scalar("Forgetting/Task", forgetting, task_idx)

    # Calculate final average accuracy by averaging the accuracy of the last task
    # over all tasks it has seen so far
    print("\nFinal accuracy across all tasks:")
    final_task_accuracies = task_wise_accuracy[-1]
    final_task_losses = task_wise_eval_loss[-1]
    tasks_seen = len(final_task_accuracies)
    final_average_accuracy = sum(final_task_accuracies) / tasks_seen
    final_average_loss = sum(final_task_losses) / tasks_seen
    print(
        f"Average accuracy: {final_average_accuracy:.4f}, Average loss: {final_average_loss:.4f}"
    )

    # Log final average metrics
    writer.add_scalar("Accuracy/FinalAverage", final_average_accuracy, 0)
    writer.add_scalar("Loss/Validation/FinalAverage", final_average_loss, 0)

    # Log hyperparameters with actual metrics instead of dummy values
    metric_dict = {
        "hparam/final_accuracy": final_average_accuracy,
        "hparam/final_loss": final_average_loss,
        "hparam/forgetting": sum(
            [task_max_accuracy[i] - final_task_accuracies[i] for i in range(tasks_seen)]
        )
        / tasks_seen,
    }
    writer.add_hparams(hparams, metric_dict)

    return final_average_accuracy, final_average_loss


def initialize_training(
    model: ContinualModel, dataset: ContinualDataset, args: Namespace
) -> tuple[list[float], list[list[float]], list[list[float]], int, int]:
    """
    Initialize training metrics and state, optionally resuming from a checkpoint.

    Args:
        model: The continual learning model
        dataset: The continual dataset
        args: Command line arguments

    Returns:
        tuple: (task_max_accuracy, task_wise_accuracy, task_wise_eval_loss, start_task, start_epoch)
    """
    tasks = dataset.N_TASKS
    task_max_accuracy = [0.0] * tasks
    task_wise_accuracy = [[0.0] * tasks for _ in range(tasks)]
    task_wise_eval_loss = [[0.0] * tasks for _ in range(tasks)]

    print(
        f"Training on {tasks} tasks with {dataset.N_CLASSES_PER_TASK} classes per task."
    )

    if args.resume_training:
        checkpoint = os.path.join(args.checkpoint_path, model.NAME + "last.pt")
        saved_data = torch.load(checkpoint)
        model.load_state_dict(saved_data["model_state_dict"])
        model.opt.load_state_dict(saved_data["optimizer_state_dict"])
        start_task = saved_data["current_task"]
        start_epoch = saved_data["epoch"]
        task_wise_accuracy = saved_data["task_wise_accuracy"]
        task_max_accuracy = saved_data["task_max_accuracy"]
        # Load eval loss if available
        task_wise_eval_loss = saved_data.get("task_wise_eval_loss", task_wise_eval_loss)

        print(f"Resuming training from task {start_task + 1}, epoch {start_epoch + 1}.")
    else:
        start_task = 0
        start_epoch = 0

    return (
        task_max_accuracy,
        task_wise_accuracy,
        task_wise_eval_loss,
        start_task,
        start_epoch,
    )
