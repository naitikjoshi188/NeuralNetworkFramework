# Custom C++ Deep Learning Framework

A highly modular, object-oriented deep learning framework implemented completely from scratch in C++17 without relying on external linear algebra or matrix processing libraries. Designed with a focus on low-level memory efficiency, mathematical validity, and structural extensibility.

## 🚀 Framework Milestones & Capabilities

- **Custom Dynamic Tensor Engine:** Built-in 2D matrix manipulation layers utilizing cache-friendly `IKJ` row-major loops to optimize CPU cache-line usage.
- **Polymorphic Architecture:** Pure virtual base-class routing (`Layer`, `Optimizer`, `Loss`) enabling seamless, plug-and-play neural configuration building blocks via an abstract `Network` container wrapper.
- **Advanced Neural Modules:** Native implementations of modern deep learning building blocks:
  - _Layers & Activations:_ Dense (Fully-Connected), ReLU, and Softmax.
  - _Loss Engines:_ Mean Squared Error (MSE) and Categorical Cross-Entropy Loss.
  - _Optimizers:_ Standard Stochastic Gradient Descent (SGD) and adaptive Adam Optimization.
- **Production Preprocessing Toolchain:** End-to-end `DataUtils` engine managing robust CSV string-stream text parsing, column-wise Min-Max feature normalization, and categorical One-Hot expansion.
- **Strict Runtime Shape Safety:** Enforced matrix-dimension matching via hardware-level assertions to guarantee algebraic validity during forward and backward passes.

## 📊 Framework Benchmarks & Verification

To verify the mathematical accuracy and scalability of this C++ engine from scratch, the framework was evaluated on standard deep learning tasks.

### 🔢 Computer Vision: MNIST Handwritten Digit Classification
* **Dataset Scale:** 60,000 Training Samples | 10,000 Independent Testing Samples
* **Topology:** DenseLayer(784 $\rightarrow$ 128) $\rightarrow$ ReLU $\rightarrow$ DenseLayer(128 $\rightarrow$ 64) $\rightarrow$ ReLU $\rightarrow$ DenseLayer(64 $\rightarrow$ 10) $\rightarrow$ Softmax
* **Optimizer:** Adam Optimizer ($\alpha = 0.003$)
* **Loss Function:** Categorical Cross-Entropy
* **Training Execution:** Mini-Batch Gradient Descent (Batch Size: 64)

#### Performance Evaluation Matrix
| Metric | Full-Batch Sandbox (Subset) | Mini-Batch Production Engine (Full File) |
| :--- | :---: | :---: |
| **Training Size** | 5,000 samples | **60,000 samples** |
| **Testing Size** | 1,000 samples | **10,000 samples** |
| **Convergence Speed** | 50 Epochs | **10 Epochs (938 steps/epoch)** |
| **Final Training Loss** | 0.0827 | **0.0268** |
| **Out-of-Sample Accuracy** | 92.20% | **97.48%** |

## 📁 Repository Blueprint

```text
.
├── include/               # Interface Definitions & Header Blueprints
│   ├── Tensor.hpp         # Core Matrix Engine Configurations
│   ├── Layer.hpp          # Virtual Base Class Layout for Layers
│   ├── DenseLayer.hpp     # Fully Connected Layer Architecture
│   ├── ActivationLayer.hpp# Non-linear activations (ReLU)
│   ├── SoftmaxLayer.hpp   # Categorical Probabilistic Output Layer
│   ├── Optimizer.hpp      # Polymorphic Optimization Interface
│   ├── AdamOptimizer.hpp  # Adaptive Gradient Tracker
│   ├── Loss.hpp           # Evaluation Function Interface
│   ├── CrossEntropyLoss.hpp# Multi-class Error Tracking
│   └── Utils.hpp          # Preprocessing Pipeline Utility Header
│
├── src/                   # Core Framework Logic Implementation
│   ├── Tensor.cpp
│   ├── DenseLayer.cpp
│   ├── SoftmaxLayer.cpp
│   ├── CrossEntropyLoss.cpp
│   └── Utils.cpp
│
├── apps/                  # Simulation Execution Binaries
│   ├── xor_main.cpp       # Non-linear Binary Logic Solver (MSE/SGD)
│   └── test_medical.cpp   # Multi-class Tabular Diagnosis Test (CE/Adam)
│
└── data/                  # Standardized Datasets Environment
    └── medical_data.csv   # Normalized Biometric Clinical Data
```
