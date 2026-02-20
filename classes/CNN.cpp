#include "./CNN.hpp"

ConvNeuralNetwork::ConvNeuralNetwork()
	: inputWidth(0), inputHeight(0), numFilters(0), filtersDepth(0),
	  kernelSize(0), convLearnRate(0.01f) {
}

ConvNeuralNetwork::ConvNeuralNetwork(const size_t imgW, const size_t imgH,
									 const size_t kSize,
									 const size_t HiddenNodes,
									 const size_t outputNodes)
	: inputWidth(imgW), inputHeight(imgH), numFilters(3), filtersDepth(1),
	  kernelSize(kSize),
	  convLearnRate(ConvNeuralNetwork::DefaultConvLearnRate) {
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
	const size_t flattenedSize = convOutW * convOutH * this->numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	this->classifier =
		NeuralNetwork(flattenedSize, HiddenNodes, outputNodes, 1);
	this->classifier.setLearnRate(ConvNeuralNetwork::DefaultConvLearnRate /
								  10.0f);
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
	  filtersDepth(filtersDepth), kernelSize(kSize),
	  convLearnRate(ConvNeuralNetwork::DefaultConvLearnRate) {
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
	const size_t flattenedSize = convOutW * convOutH * this->numFilters;
	// 3. Initialize the internal NeuralNetwork attribute
	this->classifier =
		NeuralNetwork(flattenedSize, HiddenNodes, outputNodes, hiddenLayerLen);
	this->classifier.setLearnRate(ConvNeuralNetwork::DefaultConvLearnRate /
								  10.0f);
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
	  poolArgmax(other.poolArgmax), classifier(other.classifier) {
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
		this->poolArgmax = other.poolArgmax;
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
			convResult = convResult.convolve(this->kernels[f][d]);
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
	// Logic for convolution and pooling
	const DMatrix flattenedInput = this->performConvolution(inputImage);
	// Passing NeuralNetwork attribute
	return (this->classifier.feedForward(flattenedInput.toVector()));
}

std::vector<float>
ConvNeuralNetwork::feedForward(const std::vector<float> &inputImage) const {
	// Correctly transpose image vector into image Matrix
	DMatrix imageMatrix(this->inputHeight, this->inputWidth);
	for (size_t i = 0; i < this->inputHeight; i++) {
		for (size_t j = 0; j < this->inputWidth; j++) {
			imageMatrix.setValue(i, j, inputImage[i * this->inputWidth + j]);
		}
	}
	// Passing ConvNeuralNetwork attribute
	return (this->feedForward(imageMatrix));
}

void ConvNeuralNetwork::train(const std::vector<float> &inputImage,
							  const std::vector<float> &target) {
	// Correctly transpose image vector into image Matrix
	DMatrix imageMatrix(this->inputHeight, this->inputWidth);
	for (size_t i = 0; i < this->inputHeight; i++) {
		for (size_t j = 0; j < this->inputWidth; j++) {
			imageMatrix.setValue(i, j, inputImage[i * this->inputWidth + j]);
		}
	}
	DMatrix targetMatrix(target.size(), 1);
	for (size_t i = 0; i < target.size(); i++) {
		targetMatrix(i, 0) = target[i];
	}
	this->train(imageMatrix, targetMatrix);
}

// =============================================================================
// ConvNeuralNetwork Training Method Implementations
// =============================================================================
// performConvolution  (training overload)
//
// Runs the same forward pass as the const inference version, but records every
// intermediate result needed for backprop:
//
//   featureMaps[f * filtersDepth + d]   – post-activation map at depth d for
//                                         filter f
//   preActivation[f * filtersDepth + d] – pre-activation (z = conv + bias)
//                                         at that same position
//
// poolArgmax[f] is populated here with the argmax indices from the final 2x2
// max-pooling step so that backprop can scatter gradients to the correct
// spatial positions.
//
// Returns: flattened column vector (rows = flattenedSize, cols = 1) ready to
//          be fed into classifier.train().
// -----------------------------------------------------------------------------
DMatrix ConvNeuralNetwork::performConvolution(
	const DMatrix &inputImage, std::vector<DMatrix> &featureMaps,
	std::vector<DMatrix> &preActivation) const {
	// Allocate storage: one entry per (filter x depth) combination.
	const size_t total = this->numFilters * this->filtersDepth;
	featureMaps.resize(total);
	preActivation.resize(total);

	// poolArgmax[f] will hold the argmax vector for filter f's pooling step.
	this->poolArgmax.resize(this->numFilters);

	// Collect pooled outputs (one per filter) for flattening.
	std::vector<DMatrix> pooledMaps(this->numFilters);

	for (size_t f = 0; f < this->numFilters; ++f) {
		// The "input" to depth 0 is always the original image.
		DMatrix current(inputImage);

		for (size_t d = 0; d < this->filtersDepth; ++d) {
			const size_t idx = f * this->filtersDepth + d;

			// z = conv(current, kernel) + bias  (pre-activation)
			DMatrix z = current.convolve(this->kernels[f][d]);
			z += this->kernelBiases[f][d];
			preActivation[idx] = z;

			// a = LeakyReLU(z)  (post-activation — becomes input to next depth)
			DMatrix a(z);
			a.map(LeakyReLU);

			// featureMaps[idx] holds the post-activation output of THIS depth.
			// backpropConvolution uses featureMaps[idx-1] as the layer input.
			featureMaps[idx] = a;

			current = a;
		}

		// Final 2x2 max-pooling with argmax recording for backprop.
		// `current` is the post-activation map after all depth layers.
		pooledMaps[f] = current.maxPoolingArgmax(2, this->poolArgmax[f]);
	}

	// Flatten all pooled maps into a single column vector, matching the layout
	// used during inference so the FC classifier sees the same ordering.
	std::vector<float> flat;
	flat.reserve(pooledMaps[0].getRowLength() * pooledMaps[0].getColLength() *
				 this->numFilters);

	for (size_t f = 0; f < this->numFilters; ++f) {
		for (size_t r = 0; r < pooledMaps[f].getRowLength(); ++r) {
			for (size_t c = 0; c < pooledMaps[f].getColLength(); ++c) {
				flat.push_back(pooledMaps[f](r, c));
			}
		}
	}

	// Return as a column DMatrix (rows = flattenedSize, cols = 1).
	return DMatrix(flat);
}

// -----------------------------------------------------------------------------
// train  (DMatrix overload — called internally from the vector overload)
//
// Full forward + backward pass:
//   1. Forward: CNN conv layers  ->  flatten  ->  FC classifier
//   2. Backward: classifier.train() returns the gradient w.r.t. its input
//                (the flattened CNN output).  We reshape that gradient back
//                into per-filter pool-sized error maps and call
//                backpropConvolution for each filter.
// -----------------------------------------------------------------------------
void ConvNeuralNetwork::train(const DMatrix &inputImage,
							  const DMatrix &target) {
	// ── 1. Forward pass (training variant) ──────────────────────────────────
	std::vector<DMatrix> featureMaps;
	std::vector<DMatrix> preActivation;

	const DMatrix flatInput =
		this->performConvolution(inputImage, featureMaps, preActivation);

	// ── 2. FC classifier forward + backward ─────────────────────────────────
	// classifier.train() performs a full forward+backward pass and returns the
	// gradient that should flow back into its input layer (shape: flattenedSize
	// x 1 column vector).
	const DMatrix fcGradient = this->classifier.train(flatInput, target);

	// ── 3. Reshape FC gradient back into per-filter error maps ──────────────
	// The flattened layout is [ filter-0 pixels ... | filter-1 pixels ... |
	// ...] All filters' pool outputs share the same spatial size.
	const size_t   lastD = this->filtersDepth - 1;
	const DMatrix &refPrePool = featureMaps[0 * this->filtersDepth + lastD];
	const size_t   prePoolRows = refPrePool.getRowLength();
	const size_t   prePoolCols = refPrePool.getColLength();

	// Pool output dims (2x2 pooling halves each spatial dimension).
	const size_t poolOutRows = prePoolRows / 2;
	const size_t poolOutCols = prePoolCols / 2;
	const size_t poolOutSize = poolOutRows * poolOutCols;

	const std::vector<float> gradVec = fcGradient.toVector();

	for (size_t f = 0; f < this->numFilters; ++f) {
		// Slice the gradient segment belonging to this filter.
		const size_t	   offset = f * poolOutSize;
		std::vector<float> fGradVec(gradVec.begin() + offset,
									gradVec.begin() + offset + poolOutSize);

		// Reshape from flat vector into a (poolOutRows x poolOutCols) matrix.
		DMatrix errorFromFC(poolOutRows, poolOutCols);
		for (size_t r = 0; r < poolOutRows; ++r)
			for (size_t c = 0; c < poolOutCols; ++c)
				errorFromFC(r, c) = fGradVec[r * poolOutCols + c];

		// ── 4. Backprop through conv layers for this filter ──────────────
		this->backpropConvolution(inputImage, featureMaps, preActivation,
								  errorFromFC, f);
	}
}

// -----------------------------------------------------------------------------
// backpropConvolution
//
// Backpropagates the error from the FC layer through pooling and all conv
// depth layers for a single filter `f`, then updates kernels and biases.
//
// Per depth layer (deepest -> shallowest):
//   a) Unpool: scatter the pooled gradient back to pre-pool positions via the
//              saved argmax (only for the last/deepest layer).
//   b) Apply DLeakyReLU at the saved pre-activation values (chain rule).
//   c) Kernel gradient:  dK = cross_correlate(layerInput, delta)
//      Bias gradient:    db = sum(delta)
//      Update:           K += lr * dK,  b += lr * db
//   d) Propagate delta to the previous layer via full convolution with the
//      (implicitly flipped) kernel — this is kernelMultFullPadded().
// -----------------------------------------------------------------------------
void ConvNeuralNetwork::backpropConvolution(
	const DMatrix &imageMatrix, const std::vector<DMatrix> &featureMaps,
	const std::vector<DMatrix> &preActivation, const DMatrix &errorFromFC,
	const size_t f) {
	DMatrix delta = errorFromFC;

	for (long d = static_cast<long>(this->filtersDepth) - 1; d >= 0; --d) {
		const size_t ud = static_cast<size_t>(d);
		const size_t idx = f * this->filtersDepth + ud;

		// ── a) Unpool: only for the last depth layer ──────────────────────
		// At entry, delta is in pooled (halved) space for the deepest layer.
		// maxPoolingUnpool scatters each gradient value back to the position
		// that originally won the max, and zeros out all other positions.
		if (ud == this->filtersDepth - 1) {
			const DMatrix &prePoolMap = featureMaps[idx]; // post-act, pre-pool
			delta = delta.maxPoolingUnpool(this->poolArgmax[f],
										   prePoolMap.getRowLength(),
										   prePoolMap.getColLength());
		}

		// ── b) Chain rule: multiply by derivative of activation ───────────
		// preActivation[idx] is z (before LeakyReLU).
		DMatrix actDeriv(preActivation[idx]);
		actDeriv.map(DLeakyReLU);
		delta.multiply(actDeriv); // element-wise: delta <- delta * f'(z)

		// ── c) Determine the input that was fed into this depth layer ─────
		// Depth 0 receives the original image; deeper layers receive the
		// post-activation output of the previous depth.
		const DMatrix &layerInput =
			(ud == 0) ? imageMatrix
					  : featureMaps[f * this->filtersDepth + (ud - 1)];

		// Kernel gradient = cross-correlation(layerInput, delta).
		// kernelMult() slides `delta` (treated as the "kernel") over
		// layerInput WITHOUT flipping — this is a plain cross-correlation,
		// which is exactly dL/dK for a valid-convolution forward pass.
		DMatrix kernelGrad = layerInput.convolve(delta);

		// Bias gradient = sum of all delta elements.
		const float biasGrad = delta.totalSum();

		// ── Apply learning rate and gradient clipping ─────────────────────
		kernelGrad *= this->convLearnRate;
		kernelGrad.map(clampGradient);

		// ── Update kernel and bias ────────────────────────────────────────
		this->kernels[f][ud] += kernelGrad;
		this->kernelBiases[f][ud] += this->convLearnRate * biasGrad;

		// Keep bias within a small stable range.
		float &b = this->kernelBiases[f][ud];
		if (std::isnan(b) || std::isinf(b))
			b = 0.0f;
		else if (b > 1.0f)
			b = 1.0f;
		else if (b < -1.0f)
			b = -1.0f;

		// ── d) Propagate delta to the previous depth layer ────────────────
		// The gradient w.r.t. the previous layer's output is a full
		// convolution of delta with the 180-degree-rotated kernel.
		// kernelMultFullPadded applies the kernel without flipping (plain
		// cross-correlation on full-padded input), which is mathematically
		// identical to a full convolution with the flipped kernel — the
		// correct backprop operation through a valid convolution.
		if (d > 0) {
			delta = delta.convolveFullPadded(this->kernels[f][ud]);
		}
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
		imageMatrix = imageMatrix.convolve(this->kernels[filter][d]);
		// Add bias
		imageMatrix += this->kernelBiases[filter][d];
		// Apply LeakyReLU activation
		imageMatrix.map(LeakyReLU);
	}
	// Now apply the requested kernel
	DMatrix convResult = imageMatrix.convolve(this->kernels[filter][depth]);
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
