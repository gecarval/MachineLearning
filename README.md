<div align="center">

# 🧠 MachineLearning

**Perceptrons, neural networks and CNNs written from scratch in C++ that you can watch learn.**

Draw a digit and see a convolutional network trained on MNIST read it, live.

<p>
<img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" />
<img alt="raylib" src="https://img.shields.io/badge/raylib-graphics-black?logo=raylib" />
<img alt="Dear ImGui" src="https://img.shields.io/badge/Dear%20ImGui-UI-4b8bbe" />
<img alt="MNIST" src="https://img.shields.io/badge/dataset-MNIST-orange" />
<img alt="License" src="https://img.shields.io/github/license/gecarval/MachineLearning?color=green" />
</p>
<p>
<img alt="Code size" src="https://img.shields.io/github/languages/code-size/gecarval/MachineLearning?color=lightblue" />
<img alt="Top language" src="https://img.shields.io/github/languages/top/gecarval/MachineLearning?color=blue" />
<img alt="Last commit" src="https://img.shields.io/github/last-commit/gecarval/MachineLearning?color=green" />
</p>

[Quick start](#-quick-start) · [Features](#-features) · [Controls](#-controls) · [How it works](#-how-it-works) · [Resources](#-learning-resources)

</div>

---

## ✨ Why this project

There's no PyTorch and no TensorFlow here. Every matrix multiply, convolution, pooling step and gradient is written by hand in C++, and drawn on screen while the network trains. It's an **educational reference** and a **sandbox**: read the math, change a line, and see the effect at 60 FPS.

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/gecarval/MachineLearning.git
cd MachineLearning

# 2. Build (raylib, ImGui and nlohmann/json are bundled)
make

# 3. Run
./machinelearn
```

> **Requirements:** a C++17 compiler with OpenMP (`g++`), `make`, and an OpenGL-capable desktop (Linux/X11 or Wayland).
> Extra make targets: `make debug` runs the app under Valgrind.

### Optional: export MNIST for "Train on dataset"

The CNN screen can train on MNIST PNGs from a `traindata/` folder next to the binary:

```bash
cd python_db
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt   # numpy, pillow, matplotlib
python3 image.py                             # writes traindata/0 … traindata/9
deactivate
mv traindata ../ && cd ..
```

## 🧩 Features

| | Module | What it does |
|---|---|---|
| 🔹 | **Perceptron** | Binary linear classifier that trains live on a 2D point cloud |
| 🔸 | **Neural Network** | Fully connected, configurable depth, 8 activations (Sigmoid, Tanh, ReLU, LeakyReLU, SiLU, Linear, Step, Softmax), backprop with gradient clamping, JSON save and load |
| 🔷 | **Conv Neural Network** | LeNet-style CNN with He init, 2×2 max pooling with argmax routing and a softmax classifier head. Learns MNIST digits |
| 🧮 | **DMatrix** | Hand-written matrix library: blocked matmul, valid/half/full convolution, cross-correlation, max/avg pooling |
| 🎨 | **Drawing canvas** | 28×28 grid with a Gaussian brush that looks like MNIST ink |
| 🔬 | **Filter gallery** | Look at every convolution feature map, by layer and by filter |
| 💾 | **Serialization** | Save and restore full models (weights, biases, hyperparameters) as JSON |

## 📈 Results

A headless check uses the same CNN configuration as the app: 8 filters of 5×5, 128 hidden nodes, softmax output, classifier LR 0.01, conv LR 0.001. After training on the first 10 000 MNIST images and evaluating on 1 000 test images:

| Epoch | Test accuracy |
|:-----:|:-------------:|
| 1 | **88.6 %** |
| 2 | **91.5 %** |

## 🎮 Controls

Open a mode from the main menu with its button. `Space` opens the Perceptron and `Enter` opens the Neural Network. `Esc` always returns to the menu.

<table>
<tr><th>🔷 CNN: digit recognition</th><th>🔸 Neural Network</th></tr>
<tr><td valign="top">

| Input | Action |
|---|---|
| `LMB` drag | Draw |
| `RMB` drag | Erase |
| `0`–`9` | Choose the training label |
| `C` | Clear the canvas |
| `P` | Predict (centres the digit first) |
| `T` | Train on the drawing (centres first) |
| `R` | Reset the network |
| `H` | Filter gallery (arrow keys to browse) |
| `Ctrl+S` / `Ctrl+L` | Save / load the model |
| `Ctrl+Shift+S` | Add the drawing to `traindata/` |
| `Ctrl+T` | Train on `traindata/` |
| **Epochs + / −** | Set the epochs for `Ctrl+T` |

</td><td valign="top">

| Input | Action |
|---|---|
| `LMB` | Red point (class 0) |
| `RMB` | Green point (class 1) |
| `Ctrl+Z` | Undo the last point |
| `D` | Delete all points |
| `S` / `L` | Save / load JSON |
| `R` | Reset the weights |
| `↑` / `↓` | Raise / lower the learning rate |

**🔹 Perceptron**

| Input | Action |
|---|---|
| `T` | Train on the points |
| `W A S D` | Move the view |

</td></tr>
</table>

## 🗂️ Project structure

```
MachineLearning/
├── classes/            # The ML core: no graphics code in here
│   ├── DMatrix         #   matrix and convolution library
│   ├── Perceptron      #   linear classifier
│   ├── NeuralNetwork   #   dense network + activations
│   ├── CNN             #   conv layers → pooling → NeuralNetwork head
│   └── ui/Button       #   raylib button widget
├── srcs/
│   ├── main.cpp        # window setup + network hyperparameters
│   └── states/         # one file per screen (menu, perceptron, NN, CNN)
├── includes/           # bundled raylib, Dear ImGui, nlohmann/json
└── python_db/          # MNIST IDX files + IDX → PNG exporter
```

## 📐 How it works

Click a topic to expand the derivation.

<details>
<summary><b>The Perceptron</b></summary>

The Perceptron, introduced by Frank Rosenblatt in 1958, is the simplest model of a biological neuron and the foundational building block of all neural networks.

**How it works:**

Given an input vector **x** = (x₁, x₂, …, xₙ) and a corresponding weight vector **w** = (w₁, w₂, …, wₙ) plus a bias *b*, the perceptron computes a weighted sum and passes it through a step function:

```
z = w · x + b = Σ wᵢxᵢ + b

ŷ = step(z) = { 1  if z ≥ 0
              { 0  otherwise
```

**Training — the Perceptron Learning Rule:**

For each misclassified sample, weights and bias are updated proportionally to the error:

```
wᵢ ← wᵢ + η · (y - ŷ) · xᵢ
b  ← b  + η · (y - ŷ)
```

where η is the learning rate (`LEARNRATE = 0.0000001f` in this implementation). The perceptron is **guaranteed to converge** if the data is linearly separable.

**In this project**, the Perceptron operates in 2D (each input is a `Vector2`) and can be trained interactively with keyboard controls. The decision boundary is drawn live on screen.

> **Key insight:** A single perceptron can only separate linearly separable classes. The famous failure case — XOR — cannot be solved by one perceptron. This limitation directly motivated the development of multi-layer networks.

</details>

<details>
<summary><b>Neural Networks & Backpropagation</b></summary>

A Neural Network stacks multiple layers of perceptrons (neurons), enabling it to learn non-linear decision boundaries. This implementation supports **configurable depth** (number of hidden layers), hidden size, and output size.

**Forward Pass:**

For each layer *l*, the network computes a pre-activation and then applies a nonlinearity:

```
z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f( z⁽ˡ⁾ )
```

where *f* is the chosen activation function, W⁽ˡ⁾ is the weight matrix, and b⁽ˡ⁾ is the bias vector.

**Loss:**

The network uses **mean squared error (MSE)** as the loss function:

```
L = (1/n) · Σ (yᵢ - ŷᵢ)²
```

**Backpropagation:**

Backpropagation computes the gradient of the loss with respect to every weight using the **chain rule**. Starting from the output layer and moving backward:

```
δ⁽ᴸ⁾ = ∇ₐL ⊙ f'( z⁽ᴸ⁾ )

δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾ ⊙ f'( z⁽ˡ⁾ )
```

Weights and biases are updated via **gradient descent**:

```
W⁽ˡ⁾ ← W⁽ˡ⁾ - η · δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
b⁽ˡ⁾ ← b⁽ˡ⁾ - η · δ⁽ˡ⁾
```

**Gradient safety:** This implementation includes `clampGradient()` (caps gradient magnitude at ±100) and `errorTolerance()` (zeros out gradients below 1e-4) to prevent exploding gradients and numerical instability during long training runs.

**JSON persistence:** `NeuralNetwork::serialize()` and `NeuralNetwork::deserialize()` dump/load the full weight and bias tensors to/from a JSON file, so you can save a trained network and resume later.

</details>

<details>
<summary><b>Activation Functions</b></summary>

This project implements a complete suite of activation functions and their exact analytical derivatives, used interchangeably for hidden and output layers:

| Function | Formula | Derivative | Notes |
|----------|---------|------------|-------|
| **Sigmoid** | 1 / (1 + e^−x) | σ(x)(1 − σ(x)) | Classic, saturates at extremes |
| **Tanh** | (e^x − e^−x) / (e^x + e^−x) | 1 − tanh²(x) | Zero-centered, preferred over Sigmoid |
| **ReLU** | max(0, x) | 1 if x > 0 else 0 | Sparse, fast, may "die" |
| **LeakyReLU** | x if x > 0 else x/100 | 1 or 0.01 | Fixes dying ReLU |
| **SiLU** | x · σ(x) | σ(x)(1 + x(1 − σ(x))) | Smooth, used in modern nets |
| **Linear** | x | 1 | Output layer for regression |
| **Step** | 1 if x ≥ 0 else 0 | 1 (approx) | Perceptron classifier |
| **Softmax** | e^xᵢ / Σ e^xⱼ | Jacobian | Multi-class probability output |

All functions include NaN/Inf guards and input clamping to prevent numerical blow-up during training.

**Choosing the right activation:**
- Hidden layers: **LeakyReLU** or **Tanh** work well for most tasks
- Output for binary classification: **Sigmoid**
- Output for multi-class: **Softmax**
- Output for regression: **Linear**

</details>

<details>
<summary><b>Matrix Operations (DMatrix)</b></summary>

All computations in the neural and convolutional layers are built on the custom `DMatrix` class — a row-major 2D float matrix with a rich operation set.

**Matrix multiplication** is implemented in three variants:
- `operator*` — Standard O(n³) algorithm
- `multiplyVectorized` — SIMD-friendly inner loop ordering
- `multiplyOptimized` — **Cache-blocked** multiplication with `BLOCK_SIZE = 64`, dramatically improving cache hit rates for large matrices

The cache-blocked approach divides matrices into 64×64 tiles that fit in L1/L2 cache, reducing cache misses during the innermost loop — a standard high-performance computing technique.

**Convolution & cross-correlation:**

The library provides three convolution modes (all implemented directly on `DMatrix`):

| Method | Padding | Output size | Use in CNN |
|--------|---------|-------------|-----------|
| `convolve()` | None (valid) | (H−k+1) × (W−k+1) | Shrinks feature maps |
| `convolveHalfPadded()` | Same | H × W | Preserves spatial size |
| `convolveFullPadded()` | Full | (H+k−1) × (W+k−1) | **Backprop gradient routing** |
| `kernelMult()` | None (valid) | (H−k+1) × (W−k+1) | **Forward pass cross-correlation** |

> **Note:** `kernelMult` (cross-correlation, no kernel flip) is the industry standard used by PyTorch `nn.Conv2d` and TensorFlow `Conv2D`. `convolveFullPadded` (flips the kernel) is used in backprop for mathematically correct gradient computation — see Goodfellow et al., *Deep Learning*, Ch. 9.

**Pooling:**
- `maxPooling(poolSize)` — Standard max pooling, reduces spatial size by `poolSize`
- `averagePooling(poolSize)` — Average pooling
- `maxPoolingArgmax(poolSize, argmax)` — Max pooling that **records the argmax index** of each pooling window, used during CNN training so backprop can route gradients exactly back to the winning neuron via `maxPoolingUnpool`

</details>

<details>
<summary><b>Convolutional Neural Networks</b></summary>

The `ConvNeuralNetwork` class implements a full CNN following the design of **LeCun et al. (1998)** (the original LeNet architecture), with modern conventions from PyTorch/TensorFlow.

**Architecture:**

```
Input Image (28×28, 1 channel)
        │
        ▼
┌───────────────────┐
│  Conv Layer 0     │  numFilters output channels, kernelSize×kernelSize kernels
│  + LeakyReLU      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Conv Layer 1..N  │  stacked conv layers (numConvLayers total)
│  + LeakyReLU      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2×2 Max Pooling  │  argmax recorded for backprop gradient routing
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Flatten       │  all filter maps → 1D vector
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Fully Connected  │  hiddenNodes × hiddenLayerLen → outputNodes (0–9)
│  NeuralNetwork    │
└───────────────────┘
        │
        ▼
  Class Predictions
```

**Kernel layout:** `kernels[layer][filter][inChannel]`
- Layer 0: `inChannels = 1` (grayscale image)
- Layer l > 0: `inChannels = numFilters` (output of previous layer)

**Weight initialization:** He (Kaiming) uniform — N(0, sqrt(2 / fanIn)), where `fanIn = k² × inChannels`. This initialization is specifically designed for ReLU-family activations and prevents vanishing/exploding gradients at startup.

**Forward pass per conv layer (cross-correlation):**

```
z[l][f]  =  Σ_c  crossCorrelate( input[l][c],  K[l][f][c] )  +  b[l][f]
a[l][f]  =  LeakyReLU( z[l][f] )
```

**Backpropagation through conv layers** (deepest → shallowest):

```
δ[f]               =  errorMap[f]  ⊙  DLeakyReLU( z[l][f] )

dL/dK[l][f][c]     =  crossCorrelate( input[l][c],  δ[f] )

dL/db[l][f]        =  sum( δ[f] )

dL/dinput[l][c]    =  Σ_f  fullConvolution( δ[f],  K[l][f][c] )
```

The full convolution (with kernel flip) in the input gradient is mathematically required for correct backprop through a cross-correlation forward pass.

**MNIST training canvas:**

The CNN state features a 28×28 interactive drawing canvas with a **Gaussian brush** (`BRUSH_SIGMA = 0.60f`, `BRUSH_RADIUS = 1`) that simulates the ink diffusion characteristics of MNIST handwriting samples. Draw a digit, and the CNN classifies it live. You can also visualize the intermediate feature maps produced by each conv filter at each layer.

</details>

## 📚 Learning resources

### Video Lectures

- **[3Blue1Brown — But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)** — The best visual intuition for what neural networks actually compute, with animations of how layers transform data
- **[Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=hfMk-kjRv4c)** — Building neural networks and backpropagation from scratch, in depth
- **[3Blue1Brown — Convolutions explained visually](https://www.youtube.com/watch?v=CqOfi41LfDw)** — Deep intuition for what a convolution actually does to a signal or image
- **[Andrej Karpathy — Building GPT from scratch](https://www.youtube.com/watch?v=TkwXa7Cvfr8)** — Advanced deep dive into modern neural network architectures
- **[Sebastian Lague — How Do Neural Networks Learn?](https://www.youtube.com/watch?v=3JQ3hYko51Y)** — Illustrated walkthrough of gradient descent and backpropagation
- **[3Blue1Brown — Backpropagation calculus](https://www.youtube.com/watch?v=ntKn5TPHHAk)** — Every partial derivative in the chain rule explained step by step

### Books

- **[Make Your Own Neural Network — Tariq Rashid](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608)** — Accessible, from-scratch neural network in Python; great companion to this C++ implementation
- **[The Nature of Code — Daniel Shiffman](https://natureofcode.com/)** — Evolutionary algorithms, perceptrons, and neural networks explained through generative art and simulation
- **[Deep Learning — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)** — The definitive graduate-level textbook; [Chapter 9](https://www.deeplearningbook.org/contents/convnets.html) covers convolutional networks in full mathematical rigor — used directly for the CNN backprop derivations in this project

### Articles

- **[CNNs explained — Medium (Rathna)](https://medium.com/@rathna211994/convolution-neural-networks-cnn-b6fe90214b1e)** — Practical walkthrough of CNN architecture, filters, and layer types
- **[Convolutional Neural Network from Scratch — LatinxInAI](https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07)** — Step-by-step implementation guide including forward and backward passes

### Papers

- **LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition** — The original LeNet paper that defined CNN architecture for digit recognition on MNIST
- **He et al. (2015) — Delving Deep into Rectifiers** — The paper behind He (Kaiming) weight initialization, used in this project's CNN

---

<div align="center">

Released under the [MIT License](./LICENSE) · Made by [Gerson Carvalho](https://github.com/gecarval)

</div>
