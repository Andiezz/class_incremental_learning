import torch
from models.utils.continual_model import ContinualModel
from copy import deepcopy


class ElasticWeightConsolidation(ContinualModel):
    """Elastic Weight Consolidation (EWC) method for continual learning.

    EWC adds a regularization term to the loss to prevent the model from forgetting
    previously learned tasks by constraining important parameters to stay close to
    their previous values.

    Reference:
    Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks"
    Proceedings of the National Academy of Sciences (2017)
    """

    NAME = "elastic_weight_consolidation"

    def __init__(self, backbone, loss, args, transform):
        super(ElasticWeightConsolidation, self).__init__(
            backbone, loss, args, transform
        )

        # Initialize importance measure for each parameter (Fisher Information)
        self.parameter_importance = {}

        # Initialize dictionary to store optimal parameters after each task
        self.optimal_parameters = {}

        # Regularization strength (lambda)
        self.ewc_lambda = args.ewc_lambda if hasattr(args, "ewc_lambda") else 5000

        # Current task index
        self.current_task = 0

    def compute_fisher_information(self, dataset):
        """
        Compute the Fisher Information Matrix which represents parameter importance.

        :param dataset: dataset for the current task
        """
        # Create a data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        # Initialize the parameter importance to zero
        for name, param in self.net.named_parameters():
            self.parameter_importance[name] = torch.zeros_like(param.data)

        # Set the model to evaluation mode
        self.net.eval()

        # Compute Fisher Information
        samples_so_far = 0
        num_samples = min(len(dataset), 1000)  # Limit computation to 1000 samples

        for inputs, labels in data_loader:
            if samples_so_far >= num_samples:
                break

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.shape[0]
            samples_so_far += batch_size

            # Forward pass
            outputs = self.inference(inputs)

            # For each sample in the batch
            for i in range(batch_size):
                self.net.zero_grad()
                sample_output = outputs[i].unsqueeze(0)
                sample_label = labels[i].unsqueeze(0)

                # Compute the negative log likelihood loss
                loss = self.loss(sample_output, sample_label)
                loss.backward()

                # Accumulate the gradients
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        self.parameter_importance[name] += param.grad**2 / num_samples

        # Store the current optimal parameters
        self.optimal_parameters[self.current_task] = {}
        for name, param in self.net.named_parameters():
            self.optimal_parameters[self.current_task][name] = param.data.clone()

        # Increment the task counter
        self.current_task += 1

    def training_process(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples with EWC regularization.

        :param inputs: batch of examples
        :param labels: ground-truth labels

        :return: the value of the loss function
        """
        self.net.train()
        outputs = self.inference(inputs)

        # Standard cross-entropy loss (negative log likelihood)
        loss = self.loss(outputs, labels)

        # Add EWC regularization term if not on the first task
        ewc_loss = 0

        if self.current_task > 0:
            for task_id in range(self.current_task):
                for name, param in self.net.named_parameters():
                    # Skip if parameter doesn't exist in stored parameters
                    if name not in self.optimal_parameters[task_id]:
                        continue

                    # Compute the EWC penalty
                    optimal_param = self.optimal_parameters[task_id][name]
                    importance = self.parameter_importance[name]
                    ewc_loss += torch.sum(importance * (param - optimal_param) ** 2)

            # Scale the EWC loss with lambda
            ewc_loss = self.ewc_lambda * ewc_loss
            loss += ewc_loss

        # Backpropagation and optimization
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def after_task(self, dataset):
        """
        Method called after completing a task.
        Computes the Fisher Information Matrix for the current task.

        :param dataset: dataset for the current task
        """
        self.compute_fisher_information(dataset)
