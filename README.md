# rof-mnist-resnet18
This project performs Retrieval Of Function (ROF) analysis on the MNIST dataset using a modified ResNet18 model.
## 🧾 Abstract

This project reproduces key components of the paper **"Role Taxonomy of Units in Deep Neural Networks"** by Yang Zhao, Hao Zhang, and Xiuyuan Hu (Tsinghua University), which introduces a taxonomy of neural units based on their behavior in a novel *Retrieval-of-Function (ROF)* test. Since no code was released with the paper, this implementation is constructed from the written description alone.

The authors propose categorizing units in a trained neural network by measuring their contribution to functional retrieval over both training and test sets. Units are ranked by their activation-based importance, and progressively reactivated in descending order to evaluate recovery of function (measured via classification accuracy and loss). This project implements that framework using a modified ResNet18 model trained on MNIST, and evaluates the role of each unit based on their ROF curves. Turning points in accuracy and loss are used to identify highly active units. The goal is to mirror the analysis proposed in the paper and make the methodology reproducible and testable in practice.

## Dataset

- Dataset: `MNIST` from `torchvision.datasets`
- Transformations:
  - Resize to 112×112
  - Convert to 3-channel grayscale
  - Convert to tensor
## Model
- Base model: `ResNet18`
- Final fully connected (FC) layer modified 
## Training
Optimizer: Adam
Loss function: CrossEntropyLoss
Learning rate: 0.001
Epochs: 3
Batch size: 128
Model is trained on the 5k-sample train subset
## Evaluation
Evaluation metric: classification accuracy
Evaluation is performed on the 1k-sample test subset
A baseline test accuracy is computed after training
## ROF via Feature Vector Masking

- Features are extracted from `model.avgpool` (before the fully connected layer).
- The L1-norm is computed across each feature dimension.
- Only the top-K units (based on L1-norm) are retained; all others are zero-masked.
- Evaluation is conducted over all 512 units.
- For each subset of active units, accuracy and cross-entropy loss are measured.
- Turning points are identified:
  - **n₀ₐcc**: Index where accuracy reaches maximum.
  - **n₀ₗₒₛₛ**: Index where loss reaches minimum.
- Highly active unit sets are defined as:
  - **U₀ₐcc**: Units ranked up to *n₀ₐcc*.
  - **U₀ₗₒₛₛ**: Units ranked up to *n₀ₗₒₛₛ*.
## Visualizations
Plots:
Accuracy vs. Number of Units Activated
Loss vs. Number of Units Activated
Baseline accuracy: shown with red dashed line
Turning points: shown with vertical orange lines
Output:
rof_curve_turning_points.pdf
## Baseline Analysis and ROF Evaluation

This section evaluates the baseline ResNet-18 model on the ISIC dataset and analyzes the model’s internal representations using the ROF (Ranking of Features) method.

After loading the pretrained baseline model, we extract the penultimate layer features from the test set. By ranking these units based on their L1-norm, we incrementally activate the top-k units and measure the model’s accuracy for each group (e.g., acc_00, acc_01, etc.). This process helps us understand which features contribute most to the model’s performance and how accuracy changes as more units are activated.

The results for each group’s accuracy (acc_00, acc_01, ...) are saved and visualized. For a detailed breakdown of these results, please refer to the accompanying PDF file, which contains the accuracy curves and group-wise performance metrics.

> **See:** [`rof_baseline_results.pdf`](./rof_baseline_results.pdf) for detailed accuracy plots and group analysis.
## Citation
Zhao, Y., Zhang, H., & Hu, X. (2023). Role Taxonomy of Units in Deep Neural Networks. Department of Electronic Engineering, Tsinghua University.
