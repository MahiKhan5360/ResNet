# ResNet in PyTorch

This repository contains an implementation of the **Residual Neural Network (ResNet)** architecture from scratch using PyTorch. ResNet is a powerful deep learning architecture known for its skip connections, which allow very deep networks to be trained effectively.

---

## 📌 Features

- ✅ Modular implementation of **ResNet-18**
- ✅ Customizable depth (e.g., ResNet-18, ResNet-34)
- ✅ Clean and readable PyTorch code
- ✅ Easy integration with classification pipelines

---

## 🧠 Architecture Overview

ResNet introduces **residual blocks** that add skip connections to allow gradients to flow directly through the network, mitigating the vanishing gradient problem in deep models.

****

Each residual block contains:
- 2 convolutional layers
- Batch normalization
- ReLU activation
- Skip connection

---

## 🧪 Supported Variants

| Model     | Layers Configuration        |
|-----------|-----------------------------|
| ResNet-18 | [2, 2, 2, 2]                |
| ResNet-34 | [3, 4, 6, 3]                |

More variants (e.g., ResNet-50, ResNet-101) can be implemented using `Bottleneck` blocks.

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- PyTorch >= 1.10
- torchvision

Install dependencies:

```bash
pip install torch torchvision

├── resnet.py        # ResNet-18 model implementation
├── README.md        # Documentation
