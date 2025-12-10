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

This repository contains implementations of core machine learning techniques such as the **Perceptron** algorithm and **Neural Networks** using C and C++. The project serves as an educational sandbox for experimenting with these algorithms at a low-level, with a focus on understanding their mathematical principles and practical performance.

## Features

- Implementation of Perceptron algorithms
- Custom Neural Network architectures
- Efficient matrix operations (including multiplication and backpropagation)
- Example footage and visualizations

## Evolutionary Algorithm

### What is an Evolutionary Algorithm?
Evolutionary algorithms are inspired by biological evolution processes such as selection, mutation, and crossover. They are used to optimize solutions iteratively, and in this repo may be used to explore optimal weights for neural networks.

### How it Works
Generally, the algorithm initializes a population of candidate solutions, evaluates them, and selects better solutions across generations.

## Perceptron

### How it Works
The Perceptron is a fundamental building block for neural networks. It takes several input values, multiplies them by their respective weights, adds a bias, and passes the result through an activation function (usually a step function).

### Weights and Biases
Weights determine the influence of each input feature, and the bias shifts the activation threshold. Training changes these values to minimize classification error.

**Example Usage:**  
You can run basic experiments with perceptrons using the compiled binary. See example code in the repository.

## Neural Network

### How it Works
Neural Networks are composed of layers of perceptrons ("neurons"). Inputs are processed through interconnected weights, and outputs are produced through the final layer. Training involves adjusting all weights/biases using techniques like backpropagation and gradient descent.

### Backpropagation
This algorithm computes derivatives of the error function with respect to each weight, enabling efficient update and learning.

## Multiplication Matrices

### How it Works
Matrix multiplication is central for propagating data through layers of a neural network.

### Why it's Faster
Optimized matrix operations leverage parallel computation and memory access patterns, dramatically speeding up neural network computations.

### Backpropagation
Backpropagation relies heavily on matrix operations to calculate the gradients necessary for learning.

---

# Installation

To build and run the project:

1. Clone the repository:
	```sh
	git clone https://github.com/gecarval/MachineLearning.git
	```
2. Move to the project folder:
	```sh
	cd MachineLearning
	```
3. Compile the project:
	```sh
	make
	```
4. Execute the main program:
	```sh
	./machinelearn
	```

## Controls

| BUTTON | ACTION |
|--------|--------|
|Perceptron|Press Space|
|`Esc`| Exit|
|`T`| Train|
|`W`| Up|
|`S`| Down|
|`A`| Left|
|`D`| Right|
|--------|--------|
|Neural Network|Press Enter|
|`Esc`| Exit|
|`LMB`| Red point|
|`RMB`| Green point|
|`Ctrl+Z`| Undo point|
|`D`| Delete all points|
|`S`| Save Neural Network|
|`L`| Load Neural Network|
|`R`| Reset Neural Network|
|`up`| increase learn rate|
|`down`| decrease learn rate|
