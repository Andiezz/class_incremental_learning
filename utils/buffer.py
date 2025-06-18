import torch


class Buffer:
    """
    The memory buffer of experience replay method.
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.experience_buffer = {
            "inputs": [],
            "labels": [],
        }
        self.seen_classes: dict[int, int] = {}
        self.current_task = 0
        self.class_size = 0

    def add_experience(self, inputs, labels):
        """
        Add a new experience to the buffer while maintaining equal distribution across all classes.
        When new classes arrive, redistribute buffer space to maintain balance.

        Args:
            inputs: Tensor of input samples
            labels: Tensor of corresponding labels
        """
        self.experience_buffer["inputs"].append(inputs.to(self.device))
        self.experience_buffer["labels"].append(labels.to(self.device))
        for label in labels:
            label_item = label.item()
            if label_item in self.seen_classes:
                self.seen_classes[label_item] += 1
            else:
                self.seen_classes[label_item] = 1
        self.class_size = self.buffer_size // len(self.seen_classes)

        if len(self.experience_buffer["inputs"]) > self.buffer_size:
            for class_label, size in self.seen_classes.items():
                if size > self.class_size:
                    # Remove excess samples for this class
                    indices_to_remove = [
                        i
                        for i, label in enumerate(self.experience_buffer["labels"])
                        if label.item() == class_label
                    ]
                    indices_to_remove = indices_to_remove[: size - self.class_size]
                    for index in sorted(indices_to_remove, reverse=True):
                        del self.experience_buffer["inputs"][index]
                        del self.experience_buffer["labels"][index]

    def sample_experience(self, batch_size):
        """
        Sample a batch of experiences from the buffer with balanced class distribution.
        If the buffer does not have enough samples for balanced sampling, it samples as many as possible per class and fills the rest randomly.

        Args:
            batch_size: Number of samples to return

        Returns:
            tuple: (inputs, labels) tensors of sampled experiences
        """
        current_size = len(self.experience_buffer["inputs"])
        if current_size == 0:
            raise ValueError("Buffer is empty. Cannot sample experience.")

        # Determine unique classes in the buffer
        unique_classes = list(set([label.item() if isinstance(label, torch.Tensor) else label for label in self.experience_buffer["labels"]]))
        if len(unique_classes) == 0:
            raise ValueError("No unique classes found in the buffer.")
        samples_per_class = batch_size // len(unique_classes)

        # Sample indices for each class
        sampled_indices = []
        for class_label in unique_classes:
            class_indices = [
                i
                for i, label in enumerate(self.experience_buffer["labels"])
                if (label.item() if isinstance(label, torch.Tensor) else label) == class_label
            ]

            if len(class_indices) == 0:
                continue  # Skip if no samples for this class

            num_samples = min(samples_per_class, len(class_indices))
            if num_samples > 0:
                selected = torch.randperm(len(class_indices))[:num_samples]
                sampled_indices.extend([class_indices[i] for i in selected])

        # Handle remaining samples if batch_size isn't perfectly divisible
        remaining = batch_size - len(sampled_indices)
        if remaining > 0:
            all_indices = list(set(range(current_size)) - set(sampled_indices))
            if len(all_indices) > 0:
                selected = torch.randperm(len(all_indices))[:remaining]
                sampled_indices.extend([all_indices[i] for i in selected])

        # Gather sampled experiences
        input_tensors = [self.experience_buffer["inputs"][i] for i in sampled_indices]
        try:
            inputs = torch.cat(input_tensors)
        except Exception:
            inputs = torch.stack(input_tensors)
        labels = torch.tensor(
            [self.experience_buffer["labels"][i].item() if isinstance(self.experience_buffer["labels"][i], torch.Tensor) else self.experience_buffer["labels"][i] for i in sampled_indices],
            dtype=torch.long,
            device=self.device
        )

        return inputs, labels

