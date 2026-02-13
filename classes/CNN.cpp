#include "./CNN.hpp"

ConvNeuralNetwork::ConvNeuralNetwork()
	: inputWidth(0), inputHeight(0), numFilters(0), kernelSize(0),
	  convLearnRate(0.001f) {
}

ConvNeuralNetwork::ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters,
									 size_t kSize, size_t outputNodes)
	: inputWidth(imgW), inputHeight(imgH), numFilters(filters),
	  kernelSize(kSize), convLearnRate(0.001f) {
	// 1. Initialize Kernels with random values
	for (size_t i = 0; i < numFilters; ++i) {
		DMatrix k(kernelSize, kernelSize);
		k.randomize(kernelSize * kernelSize);
		kernels.push_back(k);
		kernelBiases.push_back(0.01f);
	}
	// 2. Calculate the size of the flattened output after convolution
	// Note: Assuming "valid" padding (output is smaller than input)
	size_t convOutW = imgW - kSize + 1;
	size_t convOutH = imgH - kSize + 1;
	size_t flattenedSize = convOutW * convOutH * numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	// We use 1 hidden layer as a default here
	classifier =
		NeuralNetwork(flattenedSize, flattenedSize / 2, outputNodes, 1);
	this->classifier.setLearnRate(
		0.001f); // Set a default learning rate for the classifier
	this->classifier.enableSoftmax(
		true); // Enable softmax for multi-class classification
}

ConvNeuralNetwork::ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters,
									 size_t kSize, size_t hiddenLayerLen,
									 size_t outputNodes)
	: inputWidth(imgW), inputHeight(imgH), numFilters(filters),
	  kernelSize(kSize), convLearnRate(0.001f) {
	// 1. Initialize Kernels with random values
	for (size_t i = 0; i < numFilters; ++i) {
		DMatrix k(kernelSize, kernelSize);
		k.randomize(kernelSize * kernelSize);
		kernels.push_back(k);
		kernelBiases.push_back(0.01f);
	}
	// 2. Calculate the size of the flattened output after convolution
	// Note: Assuming "valid" padding (output is smaller than input)
	size_t convOutW = imgW - kSize + 1;
	size_t convOutH = imgH - kSize + 1;
	size_t flattenedSize = convOutW * convOutH * numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	// We use the specified number of hidden layers here
	std::cout << "CNN:" << flattenedSize << ":" << flattenedSize / 2 << ":"
			  << outputNodes << ":" << hiddenLayerLen << std::endl;
	classifier = NeuralNetwork(flattenedSize, flattenedSize / 2, outputNodes,
							   hiddenLayerLen);
	this->classifier.setLearnRate(
		0.001f); // Set a default learning rate for the classifier
	this->classifier.enableSoftmax(
		true); // Enable softmax for multi-class classification
}

// This function should apply each kernel to the input image and return a
// flattened DMatrix, along with the feature maps for backprop
DMatrix ConvNeuralNetwork::performConvolution(
	const DMatrix &inputImage, std::vector<DMatrix> &featureMaps,
	std::vector<DMatrix> &preActivation) const {
	// Calculate output dimensions for each filter
	size_t convOutW = inputWidth - kernelSize + 1;
	size_t convOutH = inputHeight - kernelSize + 1;
	// Clear and resize output vectors
	featureMaps.clear();
	preActivation.clear();
	// Apply each kernel to the input image
	for (size_t f = 0; f < numFilters; ++f) {
		DMatrix featureMap(convOutH, convOutW);
		DMatrix preAct(convOutH, convOutW);
		for (size_t i = 0; i < convOutH; ++i) {
			for (size_t j = 0; j < convOutW; ++j) {
				float sum = 0.0f;
				for (size_t ki = 0; ki < kernelSize; ++ki) {
					for (size_t kj = 0; kj < kernelSize; ++kj) {
						sum += inputImage.getValue(i + ki, j + kj) *
							   kernels[f].getValue(ki, kj);
					}
				}
				sum += kernelBiases[f]; // Add bias
				// Store pre-activation value for backprop
				preAct.setValue(i, j, sum);
				// Apply ReLU activation function: max(0, x)
				const float activated = std::max(0.0f, sum);
				featureMap.setValue(i, j, activated);
			}
		}
		featureMaps.push_back(featureMap);
		preActivation.push_back(preAct);
	}
	// After performing convolution for all filters, flatten the output
	size_t	flattenedSize = convOutW * convOutH * numFilters;
	DMatrix flattened(flattenedSize, 1);
	size_t	index = 0;
	for (const auto &featureMap : featureMaps) {
		for (size_t i = 0; i < convOutH; ++i) {
			for (size_t j = 0; j < convOutW; ++j) {
				flattened.setValue(index, 0, featureMap.getValue(i, j));
				++index;
			}
		}
	}
	return flattened;
}

// Simpler version for inference only (no backprop data needed)
DMatrix ConvNeuralNetwork::performConvolution(const DMatrix &inputImage) const {
	std::vector<DMatrix> featureMaps;
	std::vector<DMatrix> preActivation;
	return performConvolution(inputImage, featureMaps, preActivation);
}

// High-level feedforward: Convolution -> Flatten -> NeuralNetwork
std::vector<float>
ConvNeuralNetwork::feedForward(const DMatrix &inputImage) const {
	// Correctly transpose image vector into image Matrix
	DMatrix imageMatrix(this->inputHeight, this->inputWidth);
	for (size_t i = 0; i < this->inputHeight; i++) {
		for (size_t j = 0; j < this->inputWidth; j++) {
			imageMatrix.setValue(i, j, inputImage(i * this->inputWidth + j, 0));
		}
	}
	// Logic for convolution and pooling goes down here
	const DMatrix &flattenedInput = this->performConvolution(imageMatrix);
	// Passing NeuralNetwork attribute
	return classifier.feedForward(flattenedInput.toVector());
}

// Backpropagate error through convolutional layers
void ConvNeuralNetwork::backpropConvolution(
	const DMatrix &inputImage, const std::vector<DMatrix> &featureMaps,
	const std::vector<DMatrix> &preActivation, const DMatrix &errorFromFC) {
	(void)inputImage;
	(void)featureMaps;
	(void)preActivation;
	(void)errorFromFC;
}

void ConvNeuralNetwork::train(const DMatrix &inputImage,
							  const DMatrix &target) {
	(void)inputImage;
	(void)target;
}

void ConvNeuralNetwork::setClassifier(const NeuralNetwork &classifier) {
	this->classifier = classifier;
}

NeuralNetwork &ConvNeuralNetwork::getClassifier() {
	return (this->classifier);
}

const NeuralNetwork &ConvNeuralNetwork::getClassifier() const {
	return (this->classifier);
}

void ConvNeuralNetwork::setConvLearnRate(float lr) {
	this->convLearnRate = lr;
}

float ConvNeuralNetwork::getConvLearnRate(void) const {
	return this->convLearnRate;
}

const std::vector<DMatrix> &ConvNeuralNetwork::getKernels() const {
	return this->kernels;
}

const std::vector<float> &ConvNeuralNetwork::getKernelBiases() const {
	return this->kernelBiases;
}
