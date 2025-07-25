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
- Subsets used:
  - Train: 5,000 samples
  - Test: 1,000 samples
## Model
- Base model: `ResNet18`
- Final fully connected (FC) layer modified:
  ```python
  model.fc = nn.Linear(model.fc.in_features, 10)
  Device used: "mps" 
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
Features from model.avgpool are extracted (before fc)
L1-norm is computed per feature dimension
Only top-K units are kept, others zero-masked
Evaluated across all 512 units
Accuracy and cross-entropy loss are measured
Turning points are identified:
n₀_acc: index of max accuracy
n₀_loss: index of min loss
Highly active unit sets:
U₀_acc: units ranked up to n₀_acc
U₀_loss: units ranked up to n₀_loss

## Visualizations
Plots:
Accuracy vs. Number of Units Activated
Loss vs. Number of Units Activated
Baseline accuracy: shown with red dashed line
Turning points: shown with vertical orange lines
Output:
rof_curve_turning_points.pdf
## Citation
Zhao, Y., Zhang, H., & Hu, X. (2023). Role Taxonomy of Units in Deep Neural Networks. Department of Electronic Engineering, Tsinghua University.
