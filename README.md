<h2 align="center">
	REPOSITORY STATS
</h2>

<p align="center">
	<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/gecarval/MachineLearning?color=lightblue" />
	<img alt="Code language count" src="https://img.shields.io/github/languages/count/gecarval/MachineLearning?color=yellow" />
	<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/gecarval/MachineLearning?color=blue" />
	<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/gecarval/MachineLearning?color=green" />
</p>

# MachineLearning Project

This repository contains **from-scratch implementations** of core machine learning techniques — **Perceptron**, **Neural Networks**, and **Convolutional Neural Networks (CNNs)** — written in C and C++ with real-time interactive visualizations powered by [Raylib](https://www.raylib.com/) and [Dear ImGui](https://github.com/ocornut/imgui). The project is both an educational reference and a working sandbox for experimenting with machine learning algorithms at a low level, understanding their mathematical foundations, and seeing them train in real time.

The CNN is trained on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/) and can classify digits you draw yourself on a 28×28 pixel canvas — live, inside the app.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Mathematical Foundations](#mathematical-foundations)
  - [The Perceptron](#the-perceptron)
  - [Neural Networks & Backpropagation](#neural-networks--backpropagation)
  - [Activation Functions](#activation-functions)
  - [Matrix Operations (DMatrix)](#matrix-operations-dmatrix)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
- [Installation](#installation)
- [Controls](#controls)
- [MNIST & Training Data](#mnist--training-data)
- [References & Learning Resources](#references--learning-resources)

---

## Features

- **Perceptron** — Binary linear classifier with real-time training visualization on 2D point clouds
- **Fully Connected Neural Network** — Multi-layer network with configurable depth, customizable activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, SiLU, Linear, Step, Softmax), backpropagation, gradient clamping, and JSON save/load
- **Convolutional Neural Network (CNN)** — Multi-layer CNN following LeCun et al. (1998) design, trained on MNIST, with He (Kaiming) weight initialization, 2×2 max-pooling with argmax routing for backprop, and a fully connected classifier head
- **Custom DMatrix library** — Cache-optimized blocked matrix multiplication, convolution (valid, half-padded, full-padded), cross-correlation (`kernelMult`), max/average pooling, transpose, and element-wise operations
- **Interactive drawing canvas** — 28×28 grid with Gaussian brush falloff mimicking MNIST ink diffusion, for live digit inference
- **Filter visualizer** — Renders intermediate convolutional feature maps per layer and filter inside the app
- **JSON serialization** — Save and restore full network state (weights, biases, hyperparameters) to/from `.json` files
- **Python MNIST preprocessor** — Converts raw IDX binary files to organized `traindata/0–9/` PNG directories

---

## Project Structure

```
MachineLearning/
├── classes/
│   ├── Perceptron.hpp / .cpp       # Binary linear classifier
│   ├── NeuralNetwork.hpp / .cpp    # Fully connected network + activation functions
│   ├── CNN.hpp / .cpp              # Convolutional Neural Network
│   ├── DMatrix.hpp / .cpp          # Custom matrix library
│   ├── JsonParser.hpp              # JSON utility helpers
│   └── ui/
│       └── Button.hpp / .cpp       # UI button component (Raylib)
├── srcs/
│   ├── main.cpp                    # Application entry point
│   └── states/
│       ├── mainMenu.cpp            # Main menu state
│       ├── perceptronState.cpp     # Perceptron visualization state
│       ├── neuralNetworkState.cpp  # Neural Network interactive state
│       └── CNNState.cpp            # CNN training & inference state
├── includes/
│   ├── Machine.hpp                 # Global app state header
│   ├── raylib/                     # Raylib graphics library
│   └── imgui/                     # Dear ImGui + rlImGui bindings
├── python_db/
│   ├── image.py                    # MNIST IDX → PNG converter
│   ├── train-images.idx3-ubyte     # MNIST training images (60,000)
│   └── t10k-images.idx3-ubyte      # MNIST test images (10,000)
├── LICENSE
└── README.md
```

---

## Mathematical Foundations

### The Perceptron

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

---

### Neural Networks & Backpropagation

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

---

### Activation Functions

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

---

### Matrix Operations (DMatrix)

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

---

### Convolutional Neural Networks

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

---

## Installation

**Dependencies:**
- C++17 compiler (g++ or clang++)
- [Raylib](https://www.raylib.com/) (bundled as static library in `includes/raylib/`)
- [Dear ImGui](https://github.com/ocornut/imgui) + rlImGui bindings (bundled in `includes/imgui/`)
- [nlohmann/json](https://github.com/nlohmann/json) (bundled as `includes/json.hpp`)
- Python 3 + `numpy` + `Pillow` + `matplotlib` (for MNIST preprocessing only)

**Build & Run:**

```sh
# Clone the repository
git clone https://github.com/gecarval/MachineLearning.git
cd MachineLearning

# Compile
make

# Run
./machinelearn
```

**MNIST preprocessing (optional — generates PNG training images):**

```sh
cd python_db
python3 image.py
# Generates traindata/0/ through traindata/9/ with labeled PNG images
```

The `image.py` script reads the raw MNIST IDX binary format, decodes the 16-byte header (magic number, image count, rows, cols), and saves each image to `traindata/<label>/image_XXXXX.png`, organized by digit class.

---

## Controls

### Perceptron (`Space` from main menu)

| Key / Button | Action |
|---|---|
| `Space` | Enter Perceptron mode |
| `T` | Train on current point set |
| `W` / `A` / `S` / `D` | Move view (Up / Left / Down / Right) |
| `Esc` | Exit to menu |

### Neural Network (`Enter` from main menu)

| Key / Button | Action |
|---|---|
| `LMB` | Place red point (class 0) |
| `RMB` | Place green point (class 1) |
| `Ctrl+Z` | Undo last point |
| `D` | Delete all points |
| `S` | Save Neural Network to JSON |
| `L` | Load Neural Network from JSON |
| `R` | Reset Neural Network (randomize weights) |
| `↑` | Increase learning rate |
| `↓` | Decrease learning rate |
| `Esc` | Exit to menu |

### CNN (from main menu)

| Key / Button | Action |
|---|---|
| Left-click drag | Paint digit on canvas |
| Right-click drag | Erase from canvas |
| `C` | Clear canvas |
| `S` | Save CNN model to JSON |
| `L` | Load CNN model from JSON |
| `Esc` | Exit to menu |

---

## MNIST & Training Data

The MNIST database (Modified National Institute of Standards and Technology) is the canonical benchmark dataset for handwritten digit recognition. It contains:

- **60,000 training images** (`train-images.idx3-ubyte`)
- **10,000 test images** (`t10k-images.idx3-ubyte`)
- Each image is **28×28 pixels**, grayscale, with pixel values in [0, 255]
- Labels are digits 0–9

The IDX file format stores images as raw binary with a 16-byte header (magic number, count, rows, cols), followed by raw uint8 pixel data. The included `python_db/image.py` script handles parsing and extraction.

The CNN in this project uses these images as training data. Each image is normalized to [0.0, 1.0] float values before being fed into the network.

---

## References & Learning Resources

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

## License

See [LICENSE](./LICENSE) for details.
