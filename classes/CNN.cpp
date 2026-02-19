#include "./CNN.hpp"

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
		this->kernels[i].reserve(this->filtersDepth);
		this->kernelBiases[i].reserve(this->filtersDepth);
		for (size_t j = 0; j < this->filtersDepth; j++) {
			const size_t currentKernelSize = this->kernelSize;
			DMatrix		 k(currentKernelSize, currentKernelSize);
			k.randomize(currentKernelSize * currentKernelSize *
						this->filtersDepth * this->numFilters);
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
			const size_t currentKernelSize =
				(j >= this->kernelSize || this->kernelSize - j < minKSize)
					? minKSize
					: this->kernelSize - j;
			DMatrix k(currentKernelSize, currentKernelSize);
			k.randomize(currentKernelSize * currentKernelSize *
						this->filtersDepth * this->numFilters);
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
	for (size_t f = 0; f < this->numFilters; f++) {
		// Start with the input image as the initial "feature map"
		DMatrix convResult(inputImage);
		for (size_t d = 0; d < this->filtersDepth; ++d) {
			// Convolution operation
			convResult = convResult.kernelMult(this->kernels[f][d]);
			// Add biases
			convResult += this->kernelBiases[f][d];
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
	const DMatrix flattenedInput = this->performConvolution(imageMatrix);
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
	featureMaps.resize(this->numFilters * this->filtersDepth);
	preActivation.resize(this->numFilters * this->filtersDepth);
	// Apply each kernel to each feature map sequentially
	std::vector<DMatrix> filteredMaps;
	filteredMaps.reserve(this->numFilters);
	// Resize poolArgmax to hold one argmax map per filter
	this->poolArgmax.assign(this->numFilters, std::vector<size_t>());
	for (size_t f = 0; f < this->numFilters; f++) {
		// Start with the input image as the initial "feature map"
		DMatrix convResult(inputImage);
		for (size_t d = 0; d < this->filtersDepth; ++d) {
			// Convolution operation
			convResult = convResult.kernelMult(this->kernels[f][d]);
			// Add biases
			convResult += this->kernelBiases[f][d];
			// Store pre-activation for backprop
			preActivation[f * this->filtersDepth + d] = convResult;
			// Apply LeakyReLU activation
			convResult.map(LeakyReLU);
			// Store feature map for backprop
			featureMaps[f * this->filtersDepth + d] = convResult;
		}
		// Max pooling with argmax tracking
		const size_t poolSize = 2;
		const size_t prePoolRows = convResult.getRowLength();
		const size_t prePoolCols = convResult.getColLength();
		const size_t outRows = prePoolRows / poolSize;
		const size_t outCols = prePoolCols / poolSize;
		DMatrix		 pooled(outRows, outCols);
		// Store one flat index per pooled output position
		this->poolArgmax[f].resize(outRows * outCols);
		for (size_t r = 0; r < outRows; ++r) {
			for (size_t c = 0; c < outCols; ++c) {
				float  maxVal = convResult(r * poolSize, c * poolSize);
				size_t maxIdx = (r * poolSize) * prePoolCols + (c * poolSize);
				for (size_t pr = 0; pr < poolSize; ++pr) {
					for (size_t pc = 0; pc < poolSize; ++pc) {
						float val =
							convResult(r * poolSize + pr, c * poolSize + pc);
						if (val > maxVal) {
							maxVal = val;
							maxIdx = (r * poolSize + pr) * prePoolCols +
									 (c * poolSize + pc);
						}
					}
				}
				pooled(r, c) = maxVal;
				this->poolArgmax[f][r * outCols + c] = maxIdx;
			}
		}
		filteredMaps.push_back(pooled);
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
	const DMatrix		 convOutput =
		this->performConvolution(imageMatrix, featureMaps, preActivation);
	// 2. Forward pass through the NeuralNetwork
	// 3. Backpropagate error through the NeuralNetwork
	const DMatrix nnError =
		this->classifier.train(convOutput.toVector(), target);
	std::vector<DMatrix> errorsPerFilter;
	errorsPerFilter.reserve(this->numFilters);
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
	for (long f = static_cast<long>(this->numFilters) - 1; f >= 0; --f) {
		this->backpropConvolution(imageMatrix, featureMaps, preActivation,
								  errorsPerFilter[f], f);
	}
}

// Backpropagate error through convolutional layers
void ConvNeuralNetwork::backpropConvolution(
	const DMatrix &imageMatrix, const std::vector<DMatrix> &featureMaps,
	const std::vector<DMatrix> &preActivation, const DMatrix &errorFromFC,
	const size_t f) {

	DMatrix currentError = errorFromFC;

	// Propagating the error back to front
	for (long d = static_cast<long>(this->filtersDepth) - 1; d >= 0; --d) {

		// 1. dZ = Error from front * Derivative of the activation (DLeakyReLU)
		DMatrix dZ =
			preActivation[f * this->filtersDepth + static_cast<size_t>(d)];
		dZ.map(DLeakyReLU);
		// Hadamard multiplication (element by element)
		DMatrix upstreamError = currentError;
		if (d == static_cast<long>(this->filtersDepth) - 1) {
			// Argmax unpooling: route gradient only through the max position
			// that was recorded during the forward pass for filter f.
			DMatrix		 unpooled(dZ.getRowLength(), dZ.getColLength());
			const size_t outRows = currentError.getRowLength();
			const size_t outCols = currentError.getColLength();
			const size_t prePoolCols = unpooled.getColLength();
			for (size_t r = 0; r < outRows; ++r) {
				for (size_t c = 0; c < outCols; ++c) {
					// Flat index in pre-pool map where the max came from
					size_t flatIdx = this->poolArgmax[f][r * outCols + c];
					size_t pr = flatIdx / prePoolCols;
					size_t pc = flatIdx % prePoolCols;
					unpooled(pr, pc) = currentError(r, c);
				}
			}
			upstreamError = unpooled;
		}
		dZ.multiply(upstreamError);

		// 2. Define the input generated by this map (image or previous map)
		const DMatrix &layerInput =
			(d == 0) ? imageMatrix
					 : featureMaps[f * this->filtersDepth +
								   static_cast<size_t>(d) - 1];

		// 3. Gradient of the Kernel (dW)
		// Is the convolution between the input of the layer and dZ
		DMatrix dW(this->kernels[f][d].getRowLength(),
				   this->kernels[f][d].getColLength());
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

		// 4. Calculate error for previous Layer (dA_prev)
		// Only necessary if there is a previous convoluctional layer
		if (d > 0) {
			DMatrix dA_prev(layerInput.getRowLength(),
							layerInput.getColLength());
			// Convolve dZ with flipped kernel to get dA_prev
			// For each position in dZ, distribute its error to the
			// corresponding region in dA_prev
			for (size_t di = 0; di < dZ.getRowLength(); ++di) {
				for (size_t dj = 0; dj < dZ.getColLength(); ++dj) {
					float dZ_val = dZ(di, dj);
					// Distribute this error to all positions in dA_prev that
					// contributed to this output
					for (size_t m = 0; m < this->kernels[f][d].getRowLength();
						 ++m) {
						for (size_t n = 0;
							 n < this->kernels[f][d].getColLength(); ++n) {
							size_t ai = di + m;
							size_t aj = dj + n;
							// Flip the kernel (rotate 180 degrees)
							size_t km =
								this->kernels[f][d].getRowLength() - 1 - m;
							size_t kn =
								this->kernels[f][d].getColLength() - 1 - n;
							dA_prev(ai, aj) +=
								dZ_val * this->kernels[f][d](km, kn);
						}
					}
				}
			}
			currentError = dA_prev;
		}
		// 5. Update the Weights of the Kernel and Biases (Gradient Descent)
		this->kernels[f][d] -= (dW * this->convLearnRate);
		this->kernelBiases[f][d] -= (dZ.totalSum() * this->convLearnRate);
	}
}

DMatrix ConvNeuralNetwork::getConvOnAFilterFunnel(const DMatrix &inputImage,
												  const size_t	 filter,
												  const size_t	 depth) const {
	// Apply the specified kernel to the input image
	if (depth >= this->filtersDepth || filter >= this->numFilters) {
		throw std::out_of_range("Filter or depth index out of range");
	}
	// Correctly transpose image vector into image Matrix
	DMatrix imageMatrix(this->inputHeight, this->inputWidth);
	for (size_t i = 0; i < this->inputHeight; i++) {
		for (size_t j = 0; j < this->inputWidth; j++) {
			imageMatrix.setValue(i, j, inputImage(i * this->inputWidth + j, 0));
		}
	}
	// If depth > 0, we need to use the output of the previous filter as input
	for (size_t d = 0; d < depth; d++) {
		imageMatrix = imageMatrix.kernelMult(this->kernels[filter][d]);
		// Add bias
		imageMatrix += this->kernelBiases[filter][d];
		// Apply LeakyReLU activation
		imageMatrix.map(LeakyReLU);
	}
	// Now apply the requested kernel
	DMatrix convResult = imageMatrix.kernelMult(this->kernels[filter][depth]);
	// Add bias
	convResult += this->kernelBiases[filter][depth];
	// Apply LeakyReLU activation
	convResult.map(LeakyReLU);
	// Apply max pooling with pool size 2 for last depth layer
	if (depth == this->filtersDepth - 1) {
		convResult = convResult.maxPooling(2);
	}
	return (convResult);
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
