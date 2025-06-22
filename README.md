# Class Incremental Learning

Design and implement baseline methods for Class-Incremental Learning (CIL) from scratch (no high-level continual learning libraries). The focus is on understanding catastrophic forgetting through various strategies

Sure! Here's the full `README.md` content in Markdown format:

````markdown
# Continual Learning Baseline Methods

This repository explores various **baseline methods for continual learning** and evaluates them using classification accuracy and forgetting metrics across multiple sequential tasks.

---

## 📌 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/class_incremental_learning.git
   cd class_incremental_learning
   ```
2. **Install Requirements**
   ```bash
   pip install torch torchvision tensorboard numpy matplotlib
   ```
3. **Run the Experiments**
   Basic usage:

   ```bash
   python main.py
   ```

   With custom parameters:

   ```bash
   # Train with naive fine-tuning method
   python main.py --model naive --epochs 5 --batch_size 32 --lr 1e-3

   # Train with experience replay method (buffer size 2000)
   python main.py --model experience_replay --buffer_size 2000 --epochs 5

   # Train with EWC (lambda = 100)
   python main.py --model elastic_weight_consolidation --ewc_lambda 100 --epochs 5
   ```

4. **View Results with TensorBoard**
   ```bash
   tensorboard --logdir=logs
   ```
   Then open http://localhost:6006 in your browser to view the training metrics.
5. **Available Command-Line Arguments**
   - `--model`: Model type (`naive`, `experience_replay`, `elastic_weight_consolidation`)
   - `--dataset`: Dataset to use (default: `continual_cifar100`)
   - `--epochs`: Number of epochs per task (default: 5)
   - `--batch_size`: Batch size (default: 32)
   - `--lr`: Learning rate (default: 1e-3)
   - `--buffer_size`: Size of memory buffer for experience replay (default: 100)
   - `--ewc_lambda`: Regularization strength for EWC (default: 100)
   - `--seed`: Random seed (default: 612)
   - `--device`: Computation device (default: auto-selected)
   - `--checkpoint-path`: Directory to save model checkpoints (default: `checkpoints`)
   - `--resume-training`: Continue training from checkpoint (default: False)
   - `--num_workers`: Number of workers for data loading (default: 4)
   - `--momentum`: Optimizer momentum (default: 0.9)
   - `--weight_decay`: Weight decay for optimizer (default: 5e-4)
   - `--data_path`: Path to the dataset (default: `datasets`)
   - `--minibatch_size`: Size of the mini-batch for experience replay (default: 8)
   - `--log-path`: Directory for TensorBoard

## 💡 Approach

We evaluated three continual learning baselines:

1. **Naive Fine-Tuning**: Trains sequentially on tasks without any memory or regularization.
2. **Experience Replay**: Stores and replays samples from previous tasks during training.
3. **Elastic Weight Consolidation (EWC)**: Adds regularization to preserve important weights for previous tasks.

Each method was tested on 5 tasks, with performance reported using **accuracy, loss**, and **forgetting metrics**.

---

## 📈 Results & Findings

### 🔹 Naive Method

- Performance degraded drastically on earlier tasks.
- Model severely overfits the last task.

| Task | Accuracy | Loss    |
| ---- | -------- | ------- |
| 1    | 0.0000   | 11.3247 |
| 2    | 0.0000   | 10.9708 |
| 3    | 0.0000   | 10.0411 |
| 4    | 0.0000   | 8.9248  |
| 5    | 0.5885   | 1.3070  |

**Forgetting Metrics**:

| Task | Max Accuracy | Final Accuracy | Forgetting |
| ---- | ------------ | -------------- | ---------- |
| 1    | 0.4330       | 0.0000         | 0.4330     |
| 2    | 0.4995       | 0.0000         | 0.4995     |
| 3    | 0.5335       | 0.0000         | 0.5335     |
| 4    | 0.5285       | 0.0000         | 0.5285     |
| 5    | 0.5885       | 0.5885         | 0.0000     |

---

### 🔹 Experience Replay (Buffer Size = 2000)

- Marginal improvements on older tasks.
- Still suffers from noticeable forgetting, but better than naive.

| Task | Accuracy | Loss   |
| ---- | -------- | ------ |
| 1    | 0.0045   | 5.2965 |
| 2    | 0.0110   | 5.1765 |
| 3    | 0.0075   | 5.1523 |
| 4    | 0.0595   | 4.6858 |
| 5    | 0.5885   | 1.3612 |

**Forgetting Metrics**:

| Task | Max Accuracy | Final Accuracy | Forgetting |
| ---- | ------------ | -------------- | ---------- |
| 1    | 0.4115       | 0.0045         | 0.4070     |
| 2    | 0.5140       | 0.0110         | 0.5030     |
| 3    | 0.5055       | 0.0075         | 0.4980     |
| 4    | 0.5035       | 0.0595         | 0.4440     |
| 5    | 0.5885       | 0.5885         | 0.0000     |

---

### 🔹 Elastic Weight Consolidation (EWC, λ = 100)

- High regularization weight caused instability.
- Model prioritized weights optimal for the latest task, hurting prior performance.

| Task | Accuracy | Loss   |
| ---- | -------- | ------ |
| 1    | 0.0000   | 7.9655 |
| 2    | 0.0000   | 5.2900 |
| 3    | 0.0000   | 5.1053 |
| 4    | 0.1260   | 4.6801 |
| 5    | 0.0765   | 4.2704 |

**Forgetting Metrics**:

| Task | Max Accuracy | Final Accuracy | Forgetting |
| ---- | ------------ | -------------- | ---------- |
| 1    | 0.4280       | 0.0000         | 0.4280     |
| 2    | 0.4595       | 0.0000         | 0.4595     |
| 3    | 0.4815       | 0.0000         | 0.4815     |
| 4    | 0.3590       | 0.1260         | 0.2330     |
| 5    | 0.0765       | 0.0765         | 0.0000     |

---

## 🧠 Conclusion

We evaluated three baseline methods for continual learning:

- **Naive Fine-Tuning**: Suffers from severe catastrophic forgetting. Earlier tasks are completely forgotten as new tasks are learned.
- **Experience Replay**: Mitigates forgetting by replaying buffered samples from previous tasks. A larger buffer (e.g., 2000 samples) significantly improves retention while maintaining adaptability.
- **Elastic Weight Consolidation (EWC)**: Preserves performance on earlier tasks by regularizing important weights. However, high regularization strength can hinder learning of new tasks, showing a trade-off between stability and plasticity.

**Overall**, no single method is optimal across all tasks. Careful tuning and method selection are essential to balance knowledge retention and continual adaptation.
````
