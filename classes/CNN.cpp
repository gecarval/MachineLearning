#include "./CNN.hpp"
#include <vector>

ConvNeuralNetwork::ConvNeuralNetwork()
	: inputWidth(0), inputHeight(0), numFilters(0), filtersDepth(0),
	  kernelSize(0), convLearnRate(0.001f) {
}

ConvNeuralNetwork::ConvNeuralNetwork(const size_t imgW, const size_t imgH,
									 const size_t kSize,
									 const size_t HiddenNodes,
									 const size_t outputNodes)
	: inputWidth(imgW), inputHeight(imgH), numFilters(3), filtersDepth(1),
	  kernelSize(kSize), convLearnRate(0.001f) {
	// Basic validation
	if (imgW < kSize || imgH < kSize) {
		throw std::invalid_argument(
			"Image dimensions must be at least as large as kernel size");
	}
	// 1. Initialize Kernels with random values
	this->kernels.resize(this->numFilters);
	this->kernelBiases.resize(this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		this->kernels.reserve(this->filtersDepth);
		this->kernelBiases.reserve(this->filtersDepth);
		for (size_t j = 0; j < this->filtersDepth; j++) {
			const size_t currentKernelSize = this->kernelSize;
			DMatrix		 k(currentKernelSize, currentKernelSize);
			k.randomize(currentKernelSize * currentKernelSize);
			this->kernels[i].push_back(k);
			this->kernelBiases[i].push_back(0.01f);
		}
	}
	// NOTE: Assuming no padding (output is smaller than input)
	// 2. Calculate the size of the flattened output after convolution
	// and after 2x2 pooling matrix;
	long convOutW = imgW;
	long convOutH = imgH;
	for (long i = 0; i < static_cast<long>(this->filtersDepth); ++i) {
		convOutW = convOutW - this->kernels[0][i].getColLength() + 1;
		convOutH = convOutH - this->kernels[0][i].getRowLength() + 1;
	}
	convOutW /= 2;
	convOutH /= 2;
	if (convOutW <= 0 || convOutH <= 0) {
		throw std::invalid_argument("Kernel size and number of filters result "
									"in invalid output dimensions");
	}
	long flattenedSize = convOutW * convOutH * this->numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	this->classifier =
		NeuralNetwork(flattenedSize, HiddenNodes, outputNodes, 1);
	this->classifier.setLearnRate(0.001f);
	this->classifier.enableSoftmax(true);
}

ConvNeuralNetwork::ConvNeuralNetwork(const size_t imgW, const size_t imgH,
									 const size_t filters,
									 const size_t filtersDepth,
									 const size_t kSize, const size_t minKSize,
									 const size_t HiddenNodes,
									 const size_t outputNodes,
									 const size_t hiddenLayerLen)
	: inputWidth(imgW), inputHeight(imgH), numFilters(filters),
	  filtersDepth(filtersDepth), kernelSize(kSize), convLearnRate(0.001f) {
	// Basic validation
	if (kSize < minKSize) {
		throw std::invalid_argument(
			"Kernel size must be at least as large as minKSize");
	}
	if (kSize < 2) {
		throw std::runtime_error("Minimum kernel size must be greater than 1");
	}
	if (filters == 0 || filtersDepth == 0) {
		throw std::invalid_argument(
			"Number of filters and depth must be greater than 0");
	}
	if (imgW < kSize || imgH < kSize) {
		throw std::invalid_argument(
			"Image dimensions must be at least as large as kernel size");
	}
	// 1. Initialize Kernels with random values
	this->kernels.resize(this->numFilters);
	this->kernelBiases.resize(this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		this->kernels[i].reserve(this->filtersDepth);
		this->kernelBiases[i].reserve(this->filtersDepth);
		for (size_t j = 0; j < this->filtersDepth; j++) {
			const size_t currentKernelSize = this->kernelSize - j < minKSize
												 ? minKSize
												 : this->kernelSize - j;
			DMatrix		 k(currentKernelSize, currentKernelSize);
			k.randomize(currentKernelSize * currentKernelSize);
			this->kernels[i].push_back(k);
			this->kernelBiases[i].push_back(0.01f);
		}
	}
	// NOTE: Assuming no padding (output is smaller than input)
	// 2. Calculate the size of the flattened output after convolution
	// and after 2x2 pooling matrix;
	long convOutW = imgW;
	long convOutH = imgH;
	for (long i = 0; i < static_cast<long>(this->filtersDepth); ++i) {
		convOutW = convOutW - this->kernels[0][i].getColLength() + 1;
		convOutH = convOutH - this->kernels[0][i].getRowLength() + 1;
	}
	convOutW /= 2;
	convOutH /= 2;
	if (convOutW <= 0 || convOutH <= 0) {
		throw std::invalid_argument("Kernel size and number of filters result "
									"in invalid output dimensions");
	}
	long flattenedSize = convOutW * convOutH * this->numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	this->classifier =
		NeuralNetwork(flattenedSize, HiddenNodes, outputNodes, hiddenLayerLen);
	this->classifier.setLearnRate(0.001f);
	this->classifier.enableSoftmax(true);
	std::cout << "ConvNN initialized with input (" << imgW << "x" << imgH
			  << "), " << filters << " filters and," << filtersDepth
			  << "depth, kernel size " << kSize << " NeuralNetwork input size "
			  << flattenedSize << ", hidden layer nodes " << HiddenNodes
			  << ", output nodes " << outputNodes << ", hidden layers length "
			  << hiddenLayerLen << std::endl;
}

ConvNeuralNetwork::ConvNeuralNetwork(const ConvNeuralNetwork &other)
	: inputWidth(other.inputWidth), inputHeight(other.inputHeight),
	  numFilters(other.numFilters), filtersDepth(other.filtersDepth),
	  kernelSize(other.kernelSize), convLearnRate(other.convLearnRate),
	  kernels(other.kernels), kernelBiases(other.kernelBiases),
	  classifier(other.classifier) {
}

ConvNeuralNetwork &
ConvNeuralNetwork::operator=(const ConvNeuralNetwork &other) {
	if (this != &other) {
		this->inputWidth = other.inputWidth;
		this->inputHeight = other.inputHeight;
		this->numFilters = other.numFilters;
		this->filtersDepth = other.filtersDepth;
		this->kernelSize = other.kernelSize;
		this->convLearnRate = other.convLearnRate;
		this->kernels = other.kernels;
		this->kernelBiases = other.kernelBiases;
		this->classifier = other.classifier;
	}
	return (*this);
}

ConvNeuralNetwork::~ConvNeuralNetwork() {
}

// Simpler version for inference only (no backprop data needed)
// Apply each kernel to the input image and return a DMatrix
DMatrix ConvNeuralNetwork::performConvolution(const DMatrix &inputImage) const {
	// Apply each kernel to each feature map sequentially
	std::vector<DMatrix> filteredMaps;
	filteredMaps.reserve(this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		// Start with the input image as the initial "feature map"
		DMatrix convResult(inputImage);
		// Variable to store output dimensions for each filter
		size_t convOutW = this->inputWidth;
		size_t convOutH = this->inputHeight;
		for (size_t f = 0; f < this->filtersDepth; ++f) {
			// Calculate output dimensions for each filter
			convOutW = convOutW - this->kernels[i][f].getColLength() + 1;
			convOutH = convOutH - this->kernels[i][f].getRowLength() + 1;
			// Convolution operation
			convResult = convResult.kernelMult(this->kernels[i][f]);
			// Add biases
			convResult += this->kernelBiases[i][f];
			// Apply LeakyReLU activation
			convResult.map(LeakyReLU);
		}
		filteredMaps.push_back(convResult.maxPooling(2));
	}
	// Flatten the final feature map to feed into the NeuralNetwork
	std::vector<float> flatResult;
	flatResult.reserve(filteredMaps[0].getColLength() *
					   filteredMaps[0].getRowLength() * this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		for (size_t j = 0; j < filteredMaps[i].getRowLength(); j++) {
			for (size_t k = 0; k < filteredMaps[i].getColLength(); k++) {
				flatResult.push_back(filteredMaps[i](j, k));
			}
		}
	}
	return (flatResult);
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
	// Logic for convolution and pooling
	const DMatrix &flattenedInput = this->performConvolution(imageMatrix);
	// Passing NeuralNetwork attribute
	return (classifier.feedForward(flattenedInput.toVector()));
}

// This function should apply each kernel to the input image and return a
// DMatrix, along with the feature maps for backprop
DMatrix ConvNeuralNetwork::performConvolution(
	const DMatrix &inputImage, std::vector<DMatrix> &featureMaps,
	std::vector<DMatrix> &preActivation) const {
	// Clear and reserve output vectors
	featureMaps.clear();
	preActivation.clear();
	featureMaps.reserve(this->numFilters);
	preActivation.reserve(this->numFilters);
	// Apply each kernel to each feature map sequentially
	std::vector<DMatrix> filteredMaps;
	filteredMaps.reserve(this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		// Start with the input image as the initial "feature map"
		DMatrix convResult(inputImage);
		// Variable to store output dimensions for each filter
		size_t convOutW = this->inputWidth;
		size_t convOutH = this->inputHeight;
		for (size_t f = 0; f < this->filtersDepth; ++f) {
			// Calculate output dimensions for each filter
			convOutW = convOutW - this->kernels[i][f].getColLength() + 1;
			convOutH = convOutH - this->kernels[i][f].getRowLength() + 1;
			// Convolution operation
			convResult = convResult.kernelMult(this->kernels[i][f]);
			// Add biases
			convResult += this->kernelBiases[i][f];
			// Store pre-activation for backprop
			preActivation.push_back(convResult);
			// Apply LeakyReLU activation
			convResult.map(LeakyReLU);
			// Store feature map for backprop
			featureMaps.push_back(convResult);
		}
		filteredMaps.push_back(convResult.maxPooling(2));
	}
	// Flatten the final feature map to feed into the NeuralNetwork
	std::vector<float> flatResult;
	flatResult.reserve(filteredMaps[0].getColLength() *
					   filteredMaps[0].getRowLength() * this->numFilters);
	for (size_t i = 0; i < this->numFilters; i++) {
		for (size_t j = 0; j < filteredMaps[i].getRowLength(); j++) {
			for (size_t k = 0; k < filteredMaps[i].getColLength(); k++) {
				flatResult.push_back(filteredMaps[i](j, k));
			}
		}
	}
	return (flatResult);
}

void ConvNeuralNetwork::train(const DMatrix &inputImage,
							  const DMatrix &target) {
	// Correctly transpose image vector into image Matrix
	DMatrix imageMatrix(this->inputHeight, this->inputWidth);
	for (size_t i = 0; i < this->inputHeight; i++) {
		for (size_t j = 0; j < this->inputWidth; j++) {
			imageMatrix.setValue(i, j, inputImage(i * this->inputWidth + j, 0));
		}
	}
	// 1. Forward pass through convolutional layers
	std::vector<DMatrix> featureMaps;
	std::vector<DMatrix> preActivation;
	const DMatrix		&convOutput =
		this->performConvolution(imageMatrix, featureMaps, preActivation);
	// 2. Forward pass through the NeuralNetwork
	const std::vector<float> &nnOutput =
		this->classifier.feedForward(convOutput.toVector());
	// 3. Backpropagate error through the NeuralNetwork
	const DMatrix &nnError =
		this->classifier.train(convOutput.toVector(), target);
	std::vector<DMatrix> errorsPerFilter;
	errorsPerFilter.reserve(this->filtersDepth);
	long convOutW = this->inputWidth;
	long convOutH = this->inputHeight;
	for (long i = 0; i < static_cast<long>(this->filtersDepth); ++i) {
		convOutW = convOutW - this->kernels[0][i].getColLength() + 1;
		convOutH = convOutH - this->kernels[0][i].getRowLength() + 1;
	}
	const size_t mapH = convOutH / 2; // The height after pooling
	const size_t mapW = convOutW / 2; // The width after pooling
	// 4. Convert error from NeuralNetwork back to convolutional layer format
	// for each filter
	for (size_t i = 0; i < this->numFilters; i++) {
		DMatrix e(mapH, mapW);
		for (size_t r = 0; r < mapH; r++) {
			for (size_t c = 0; c < mapW; c++) {
				e(r, c) = nnError(i * (mapH * mapW) + (r * mapW + c), 0);
			}
		}
		errorsPerFilter.push_back(e);
	}
	// 5. Backpropagate error through convolutional layers
	for (long i = this->numFilters; i >= 0; i++) {
		this->backpropConvolution(imageMatrix, featureMaps, preActivation,
								  errorsPerFilter[i], i);
	}
}

// Backpropagate error through convolutional layers
void ConvNeuralNetwork::backpropConvolution(
	const DMatrix &imageMatrix, const std::vector<DMatrix> &featureMaps,
	const std::vector<DMatrix> &preActivation, const DMatrix &errorFromFC,
	const size_t i) {

	DMatrix currentError = errorFromFC;

	// Percorrer os filtros de trás para frente
	for (long f = static_cast<long>(this->numFilters) - 1; f >= 0; --f) {

		// 1. dZ = Erro vindo da frente * Derivada da Ativação (DLeakyReLU)
		DMatrix dZ = preActivation[f];
		dZ.map(DLeakyReLU);
		// Multiplicação Hadamard (elemento a elemento)
		dZ.multiply(currentError);

		// 2. Definir o input que gerou este mapa (imagem ou mapa anterior)
		const DMatrix &layerInput = (f == 0) ? imageMatrix : featureMaps[f - 1];

		// 3. Gradiente do Kernel (dW)
		// É a convolução entre o Input da camada e o dZ
		DMatrix dW(this->kernels[i][f].getRowLength(),
				   this->kernels[i][f].getColLength());
		for (size_t m = 0; m < dW.getRowLength(); ++m) {
			for (size_t n = 0; n < dW.getColLength(); ++n) {
				float sum = 0;
				for (size_t i = 0; i < dZ.getRowLength(); ++i) {
					for (size_t j = 0; j < dZ.getColLength(); ++j) {
						sum += layerInput(i + m, j + n) * dZ(i, j);
					}
				}
				dW(m, n) = sum;
			}
		}

		// 4. Calcular erro para a camada anterior (dA_prev)
		// Necessário apenas se houver uma camada de convolução anterior (f > 0)
		if (f > 0) {
			DMatrix dA_prev(layerInput.getRowLength(),
							layerInput.getColLength());
			// Initialize to zero
			for (size_t i = 0; i < dA_prev.getRowLength(); ++i) {
				for (size_t j = 0; j < dA_prev.getColLength(); ++j) {
					dA_prev(i, j) = 0.0f;
				}
			}

			// Convolve dZ with flipped kernel to get dA_prev
			// For each position in dZ, distribute its error to the
			// corresponding region in dA_prev
			for (size_t di = 0; di < dZ.getRowLength(); ++di) {
				for (size_t dj = 0; dj < dZ.getColLength(); ++dj) {
					float dZ_val = dZ(di, dj);
					// Distribute this error to all positions in dA_prev that
					// contributed to this output
					for (size_t m = 0; m < this->kernels[i][f].getRowLength();
						 ++m) {
						for (size_t n = 0;
							 n < this->kernels[i][f].getColLength(); ++n) {
							size_t ai = di + m;
							size_t aj = dj + n;
							// Flip the kernel (rotate 180 degrees)
							size_t km =
								this->kernels[i][f].getRowLength() - 1 - m;
							size_t kn =
								this->kernels[i][f].getColLength() - 1 - n;
							dA_prev(ai, aj) +=
								dZ_val * this->kernels[i][f](km, kn);
						}
					}
				}
			}
			currentError = dA_prev;
		}

		// 5. Atualizar os pesos do Kernel e Bias (Gradient Descent)
		this->kernels[i][f] -= (dW * this->convLearnRate);
		this->kernelBiases[i][f] -= (dZ.totalSum() * this->convLearnRate);
	}
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
	return (this->convLearnRate);
}

size_t ConvNeuralNetwork::getInputWidth() const {
	return (this->inputWidth);
}

size_t ConvNeuralNetwork::getInputHeight() const {
	return (this->inputHeight);
}

size_t ConvNeuralNetwork::getNumFilters() const {
	return (this->numFilters);
}

size_t ConvNeuralNetwork::getFiltersDepth() const {
	return (this->filtersDepth);
}

size_t ConvNeuralNetwork::getKernelSize() const {
	return (this->kernelSize);
}

const std::vector<std::vector<DMatrix>> &ConvNeuralNetwork::getKernels() const {
	return (this->kernels);
}

const std::vector<std::vector<float>> &
ConvNeuralNetwork::getKernelBiases() const {
	return (this->kernelBiases);
}
