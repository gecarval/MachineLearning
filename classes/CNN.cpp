#include "./CNN.hpp"

// =============================================================================
// Internal helpers
// =============================================================================

size_t ConvNeuralNetwork::inChannelsAt(size_t layer) const noexcept {
	return (layer == 0) ? 1 : this->numFilters;
}

// Each valid-convolution layer (no padding) reduces the spatial side by
// (kernelSize - 1).  After `layer+1` such layers:
//   side = inputSide - (layer+1) * (kernelSize - 1)
size_t ConvNeuralNetwork::spatialSizeAfterConv(size_t layer) const noexcept {
	const long side =
		static_cast<long>(this->inputWidth) -
		static_cast<long>(layer + 1) * static_cast<long>(this->kernelSize - 1);
	return (side > 0) ? static_cast<size_t>(side) : 0;
}

// After all conv layers, the last pool halves the spatial dimensions.
size_t ConvNeuralNetwork::computeFlattenedSize() const {
	const size_t lastLayer = this->numConvLayers - 1;
	const size_t convSide = this->spatialSizeAfterConv(lastLayer);
	const size_t poolSide = convSide / 2;
	return poolSide * poolSide * this->numFilters;
}

// He (Kaiming) initialisation: W ~ N(0, sqrt(2 / fanIn))
// fanIn = kernelSize² × inChannels(layer)
void ConvNeuralNetwork::initKernels() {
	this->kernels.resize(this->numConvLayers);
	this->kernelBiases.resize(this->numConvLayers);

	for (size_t l = 0; l < this->numConvLayers; ++l) {
		const size_t nIn = this->inChannelsAt(l);
		const size_t fanIn = this->kernelSize * this->kernelSize * nIn;

		this->kernels[l].resize(this->numFilters);
		this->kernelBiases[l].assign(this->numFilters,
									 0.0f); // biases start at 0

		for (size_t f = 0; f < this->numFilters; ++f) {
			this->kernels[l][f].resize(nIn);
			for (size_t c = 0; c < nIn; ++c) {
				DMatrix k(this->kernelSize, this->kernelSize);
				k.randomize(fanIn); // He init inside DMatrix::randomize(fanIn)
				this->kernels[l][f][c] = std::move(k);
			}
		}
	}
}

DMatrix ConvNeuralNetwork::vectorToImage(const std::vector<float> &v) const {
	DMatrix img(this->inputHeight, this->inputWidth);
	for (size_t r = 0; r < this->inputHeight; ++r)
		for (size_t c = 0; c < this->inputWidth; ++c)
			img(r, c) = v[r * this->inputWidth + c];
	return img;
}

// =============================================================================
// Constructors / destructor
// =============================================================================

ConvNeuralNetwork::ConvNeuralNetwork()
	: inputWidth(0), inputHeight(0), numFilters(0), numConvLayers(0),
	  kernelSize(0), convLearnRate(DefaultConvLearnRate) {
}

ConvNeuralNetwork::~ConvNeuralNetwork() {
}

ConvNeuralNetwork::ConvNeuralNetwork(const ConvNeuralNetwork &o)
	: inputWidth(o.inputWidth), inputHeight(o.inputHeight),
	  numFilters(o.numFilters), numConvLayers(o.numConvLayers),
	  kernelSize(o.kernelSize), convLearnRate(o.convLearnRate),
	  kernels(o.kernels), kernelBiases(o.kernelBiases),
	  poolArgmax(o.poolArgmax), classifier(o.classifier) {
}

ConvNeuralNetwork &ConvNeuralNetwork::operator=(const ConvNeuralNetwork &o) {
	if (this != &o) {
		this->inputWidth = o.inputWidth;
		this->inputHeight = o.inputHeight;
		this->numFilters = o.numFilters;
		this->numConvLayers = o.numConvLayers;
		this->kernelSize = o.kernelSize;
		this->convLearnRate = o.convLearnRate;
		this->kernels = o.kernels;
		this->kernelBiases = o.kernelBiases;
		this->poolArgmax = o.poolArgmax;
		this->classifier = o.classifier;
	}
	return *this;
}

// ---------------------------------------------------------------------------
// Simple constructor: 1 conv layer, 3 filters.
// ---------------------------------------------------------------------------
ConvNeuralNetwork::ConvNeuralNetwork(size_t imgW, size_t imgH, size_t kSize,
									 size_t hiddenNodes, size_t outputNodes)
	: inputWidth(imgW), inputHeight(imgH), numFilters(3), numConvLayers(1),
	  kernelSize(kSize), convLearnRate(DefaultConvLearnRate) {

	if (imgW < kSize || imgH < kSize) {
		throw std::invalid_argument("Image dimensions must be >= kernel size");
	}
	if (kSize < 2) {
		throw std::invalid_argument("Kernel size must be >= 2");
	}
	initKernels();

	const size_t flatSize = computeFlattenedSize();
	if (flatSize == 0)
		throw std::invalid_argument(
			"Kernel size produces zero-size output after pooling");

	this->classifier = NeuralNetwork(flatSize, hiddenNodes, outputNodes, 1);
	this->classifier.setLearnRate(DefaultConvLearnRate / 10.0f);
	this->classifier.enableSoftmax(true);
}

// ---------------------------------------------------------------------------
// Full constructor.
// ---------------------------------------------------------------------------
ConvNeuralNetwork::ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters,
									 size_t numConvLayers, size_t kSize,
									 size_t hiddenNodes, size_t outputNodes,
									 size_t hiddenLayerLen)
	: inputWidth(imgW), inputHeight(imgH), numFilters(filters),
	  numConvLayers(numConvLayers), kernelSize(kSize),
	  convLearnRate(DefaultConvLearnRate) {

	if (imgW < kSize || imgH < kSize)
		throw std::invalid_argument("Image dimensions must be >= kernel size");
	if (kSize < 2) throw std::invalid_argument("Kernel size must be >= 2");
	if (filters == 0)
		throw std::invalid_argument("Number of filters must be > 0");
	if (numConvLayers == 0)
		throw std::invalid_argument("Number of conv layers must be > 0");

	// Verify that the spatial size remains positive through all layers + pool.
	for (size_t l = 0; l < numConvLayers; ++l) {
		if (spatialSizeAfterConv(l) == 0)
			throw std::invalid_argument(
				"Kernel size / depth combination produces a zero-size "
				"feature map at layer " +
				std::to_string(l));
	}

	initKernels();

	const size_t flatSize = computeFlattenedSize();
	if (flatSize == 0)
		throw std::invalid_argument(
			"Configuration produces zero-size flattened vector");

	this->classifier =
		NeuralNetwork(flatSize, hiddenNodes, outputNodes, hiddenLayerLen);
	this->classifier.setLearnRate(DefaultConvLearnRate / 10.0f);
	this->classifier.enableSoftmax(true);

	std::cout << "[ConvNN] " << imgW << "x" << imgH << " input | "
			  << numConvLayers << " conv layers | " << filters << " filters | "
			  << kSize << "x" << kSize << " kernels | "
			  << "FC input " << flatSize << " | " << hiddenNodes << " hidden | "
			  << outputNodes << " outputs | " << hiddenLayerLen
			  << " hidden layers\n";
}

// =============================================================================
// forwardConv  (inference — no intermediate storage)
//
// Runs the full conv pipeline:
//   for each layer l:
//     for each output filter f:
//       z[f] = Σ_c  kernelMult(input[c], K[l][f][c])  +  b[l][f]
//       a[f] = LeakyReLU(z[f])
//   max-pool (2×2) on each of the final layer's maps
//   flatten all pooled maps → column DMatrix
// =============================================================================
DMatrix ConvNeuralNetwork::forwardConv(const DMatrix &image) const {
	// currentMaps[c] holds the feature map for input channel c of the next
	// layer. At layer 0, this is just the single-channel input image.
	std::vector<DMatrix> currentMaps = {image};

	for (size_t l = 0; l < this->numConvLayers; ++l) {
		const size_t		 nIn = this->inChannelsAt(l);
		std::vector<DMatrix> nextMaps(this->numFilters);

		for (size_t f = 0; f < this->numFilters; ++f) {
			// z = Σ_c crossCorrelate(input[c], K[l][f][c]) + b[l][f]
			// kernelMult() = cross-correlation, no kernel flip (industry
			// standard).
			DMatrix z = currentMaps[0].kernelMult(this->kernels[l][f][0]);
			for (size_t c = 1; c < nIn; ++c)
				z += currentMaps[c].kernelMult(this->kernels[l][f][c]);
			z += this->kernelBiases[l][f];

			// a = LeakyReLU(z)
			z.map(LeakyReLU);
			nextMaps[f] = std::move(z);
		}
		currentMaps = std::move(nextMaps);
	}

	// 2×2 max-pool on every output map of the last conv layer, then flatten.
	std::vector<float> flat;
	for (size_t f = 0; f < this->numFilters; ++f) {
		DMatrix		 pooled = currentMaps[f].maxPooling(2);
		const size_t outR = pooled.getRowLength();
		const size_t outC = pooled.getColLength();
		for (size_t r = 0; r < outR; ++r)
			for (size_t c = 0; c < outC; ++c) flat.push_back(pooled(r, c));
	}
	return DMatrix(flat); // column vector (flatSize × 1)
}

// =============================================================================
// forwardConvTrain  (training — records intermediates for backprop)
//
// Identical computation to forwardConv, additionally stores:
//   layerInputs[l]  — input channel maps for layer l
//   preAct[l][f]    — z (before activation) at (l, f)
//   poolArgmax[f]   — argmax indices from final 2×2 max-pool
// =============================================================================
DMatrix ConvNeuralNetwork::forwardConvTrain(
	const DMatrix &image, std::vector<std::vector<DMatrix>> &layerInputs,
	std::vector<std::vector<DMatrix>> &preAct) {

	layerInputs.resize(this->numConvLayers);
	preAct.resize(this->numConvLayers, std::vector<DMatrix>(this->numFilters));
	this->poolArgmax.resize(this->numFilters);

	std::vector<DMatrix> currentMaps = {image};

	for (size_t l = 0; l < this->numConvLayers; ++l) {
		const size_t nIn = this->inChannelsAt(l);

		// Save the input maps for this layer (used in backprop for dL/dK).
		layerInputs[l] = currentMaps;

		std::vector<DMatrix> nextMaps(this->numFilters);

		for (size_t f = 0; f < this->numFilters; ++f) {
			// z = Σ_c crossCorrelate(input[c], K[l][f][c]) + b[l][f]
			DMatrix z = currentMaps[0].kernelMult(this->kernels[l][f][0]);
			for (size_t c = 1; c < nIn; ++c)
				z += currentMaps[c].kernelMult(this->kernels[l][f][c]);
			z += this->kernelBiases[l][f];

			preAct[l][f] = z; // save pre-activation for backprop chain rule

			// a = LeakyReLU(z)
			DMatrix a(z);
			a.map(LeakyReLU);
			nextMaps[f] = std::move(a);
		}
		currentMaps = std::move(nextMaps);
	}

	// 2×2 max-pool with argmax recording on the last layer's maps, then
	// flatten.
	std::vector<float> flat;
	for (size_t f = 0; f < this->numFilters; ++f) {
		DMatrix pooled =
			currentMaps[f].maxPoolingArgmax(2, this->poolArgmax[f]);
		const size_t outR = pooled.getRowLength();
		const size_t outC = pooled.getColLength();
		for (size_t r = 0; r < outR; ++r)
			for (size_t c = 0; c < outC; ++c) flat.push_back(pooled(r, c));
	}
	return DMatrix(flat); // column vector (flatSize × 1)
}

// =============================================================================
// backwardConv
//
// Receives `fcGrad` — the gradient returned by classifier.train() w.r.t. its
// flat input (shape: flatSize × 1).
//
// Step 1 — Unpool: scatter each gradient value back to the position that
//   won the max during the forward pass (all other positions receive 0).
//   Result: one error map per filter, same spatial size as the last conv
//   layer's post-activation output.
//
// Step 2 — For each conv layer l (from last to first):
//   For each output filter f:
//     a) Chain rule through activation:
//          delta[f] = errorMap[f]  ⊙  DLeakyReLU( preAct[l][f] )
//
//     b) Kernel gradient (cross-correlation, no flip):
//          dL/dK[l][f][c] = crossCorrelate( layerInput[c],  delta[f] )
//                         = layerInput[c].kernelMult( delta[f] )
//
//     c) Bias gradient (sum of delta, clamped before scaling):
//          dL/db[l][f] = clamp( sum( delta[f] ) )
//
//     d) Gradient w.r.t. this layer's input (to propagate further back):
//          dL/dinput[c] += fullConvolution( delta[f],  K[l][f][c] )
//                        = delta[f].convolveFullPadded( K[l][f][c] )
//          (convolveFullPadded flips the kernel — correct for cross-corr fwd)
// =============================================================================
void ConvNeuralNetwork::backwardConv(
	const DMatrix &fcGrad, std::vector<std::vector<DMatrix>> &layerInputs,
	std::vector<std::vector<DMatrix>> &preAct) {

	// ── Step 1: Unpool
	// ──────────────────────────────────────────────────────── The last conv
	// layer's post-activation maps have spatial size convSide×convSide.
	const size_t lastLayer = this->numConvLayers - 1;
	const size_t convSide = this->spatialSizeAfterConv(lastLayer);
	const size_t poolSide = convSide / 2;
	const size_t poolArea = poolSide * poolSide;

	const std::vector<float> gradVec = fcGrad.toVector();

	// errorMaps[f]: gradient scattered back to pre-pool spatial size
	// (convSide²).
	std::vector<DMatrix> errorMaps(this->numFilters);
	for (size_t f = 0; f < this->numFilters; ++f) {
		// Slice the flat gradient for this filter.
		const size_t offset = f * poolArea;
		DMatrix		 poolGrad(poolSide, poolSide);
		for (size_t r = 0; r < poolSide; ++r)
			for (size_t c = 0; c < poolSide; ++c)
				poolGrad(r, c) = gradVec[offset + r * poolSide + c];

		// Scatter via saved argmax → zeros everywhere except the max position.
		errorMaps[f] =
			poolGrad.maxPoolingUnpool(this->poolArgmax[f], convSide, convSide);
	}

	// ── Step 2: Propagate through conv layers (deepest → shallowest) ─────────
	for (long l = static_cast<long>(this->numConvLayers) - 1; l >= 0; --l) {
		const size_t ul = static_cast<size_t>(l);
		const size_t nIn = this->inChannelsAt(ul);

		// Accumulate input-gradient for every input channel of this layer.
		// (For layer 0 this is the gradient w.r.t. the raw image — discarded.)
		std::vector<DMatrix> inputGrad(nIn);

		for (size_t f = 0; f < this->numFilters; ++f) {
			// ── a) Chain rule through LeakyReLU ──────────────────────────────
			// delta = errorMap[f]  ⊙  DLeakyReLU( z[l][f] )
			DMatrix deriv(preAct[ul][f]);
			deriv.map(DLeakyReLU);
			DMatrix delta(errorMaps[f]);
			delta.multiply(deriv); // element-wise product

			for (size_t c = 0; c < nIn; ++c) {
				// ── b) Kernel gradient
				// ──────────────────────────────────────── dL/dK[l][f][c] =
				// crossCorrelate( input[c], delta ) kernelMult performs
				// cross-correlation without flipping, which is the correct
				// formula for a cross-corr forward pass.
				DMatrix kGrad = layerInputs[ul][c].kernelMult(delta);

				// ── c) Bias gradient
				// ────────────────────────────────────────── dL/db[l][f] =
				// sum(delta), clamped before scaling.
				const float bGrad = clampGradient(delta.totalSum());

				// ── Apply learning rate and gradient clipping, then update
				// ────
				kGrad *= this->convLearnRate;
				kGrad.map(clampGradient);
				this->kernels[ul][f][c] += kGrad;

				float &b = this->kernelBiases[ul][f];
				b += this->convLearnRate * bGrad;
				// Keep bias numerically stable.
				if (std::isnan(b) || std::isinf(b)) b = 0.0f;
				b = std::max(-1.0f, std::min(1.0f, b));

				// ── d) Input gradient
				// ───────────────────────────────────────── dL/dinput[c] +=
				// fullConvolution( delta, K[l][f][c] ) convolveFullPadded flips
				// the kernel (true full convolution), which is the
				// mathematically correct gradient w.r.t. the input of a
				// cross-correlation forward pass. (Goodfellow et al., Deep
				// Learning, Ch. 9; PyTorch ATen source)
				if (l > 0) {
					DMatrix ig =
						delta.convolveFullPadded(this->kernels[ul][f][c]);
					if (inputGrad[c].getRowLength() == 0)
						inputGrad[c] = std::move(ig);
					else
						inputGrad[c] += ig;
				}
			}
		}

		// The input gradient for this layer becomes the error map for the
		// previous layer.
		if (l > 0) errorMaps = std::move(inputGrad);
	}
}

// =============================================================================
// Public inference / training
// =============================================================================

std::vector<float>
ConvNeuralNetwork::feedForward(const std::vector<float> &image) const {
	return this->classifier.feedForward(
		this->forwardConv(this->vectorToImage(image)).toVector());
}

void ConvNeuralNetwork::train(const std::vector<float> &image,
							  const std::vector<float> &target) {
	const DMatrix img = this->vectorToImage(image);

	// Build target column vector.
	DMatrix tgt(target.size(), 1);
	for (size_t i = 0; i < target.size(); ++i) tgt(i, 0) = target[i];

	// ── Forward pass
	// ──────────────────────────────────────────────────────────
	std::vector<std::vector<DMatrix>> layerInputs, preAct;
	const DMatrix flatInput = this->forwardConvTrain(img, layerInputs, preAct);

	// ── FC forward + backward
	// ───────────────────────────────────────────────── classifier.train()
	// returns the gradient w.r.t. its input (flatSize × 1).
	const DMatrix fcGrad = this->classifier.train(flatInput, tgt);

	// ── Conv backward
	// ─────────────────────────────────────────────────────────
	this->backwardConv(fcGrad, layerInputs, preAct);
}

// =============================================================================
// getConvOnAFilterFunnel
//
// Runs conv layers 0…layer for the requested filter index and returns the
// resulting feature map (max-pooled if it is the last layer).
// Each intermediate layer produces only the one filter's output; that single
// map is used as the sole input channel for the next layer iteration.
// This is a visualisation / debugging helper, not used in normal training.
// =============================================================================
DMatrix ConvNeuralNetwork::getConvOnAFilterFunnel(const DMatrix &image,
												  size_t		 filter,
												  size_t		 layer) const {
	if (filter >= this->numFilters)
		throw std::out_of_range("Filter index out of range");
	if (layer >= this->numConvLayers)
		throw std::out_of_range("Layer index out of range");

	// Rebuild the image matrix from the flat input column vector.
	DMatrix img(this->inputHeight, this->inputWidth);
	for (size_t r = 0; r < this->inputHeight; ++r)
		for (size_t c = 0; c < this->inputWidth; ++c)
			img(r, c) = image(r * this->inputWidth + c, 0);

	// Run only the requested filter through each layer.
	// At layer 0 the input is the single-channel image.
	// At layer l > 0 we only have the previous iteration's single output map.
	std::vector<DMatrix> currentMaps = {img};

	for (size_t l = 0; l <= layer; ++l) {
		const size_t nIn = this->inChannelsAt(l);

		DMatrix z = currentMaps[0].kernelMult(this->kernels[l][filter][0]);
		for (size_t c = 1; c < nIn && c < currentMaps.size(); ++c)
			z += currentMaps[c].kernelMult(this->kernels[l][filter][c]);
		z += this->kernelBiases[l][filter];
		z.map(LeakyReLU);

		currentMaps = {std::move(z)};
	}

	// Apply max-pool only on the last requested layer.
	if (layer == this->numConvLayers - 1) return currentMaps[0].maxPooling(2);
	return currentMaps[0];
}

// =============================================================================
// Accessors / setters
// =============================================================================

void ConvNeuralNetwork::setClassifier(const NeuralNetwork &c) {
	this->classifier = c;
}
NeuralNetwork &ConvNeuralNetwork::getClassifier() {
	return this->classifier;
}
const NeuralNetwork &ConvNeuralNetwork::getClassifier() const {
	return this->classifier;
}

void ConvNeuralNetwork::setConvLearnRate(float lr) {
	this->convLearnRate = lr;
}
float ConvNeuralNetwork::getConvLearnRate() const {
	return this->convLearnRate;
}

size_t ConvNeuralNetwork::getInputWidth() const {
	return this->inputWidth;
}

size_t ConvNeuralNetwork::getInputHeight() const {
	return this->inputHeight;
}

size_t ConvNeuralNetwork::getNumFilters() const {
	return this->numFilters;
}

size_t ConvNeuralNetwork::getNumConvLayers() const {
	return this->numConvLayers;
}

size_t ConvNeuralNetwork::getKernelSize() const {
	return this->kernelSize;
}

const std::vector<std::vector<std::vector<DMatrix>>> &
ConvNeuralNetwork::getKernels() const {
	return this->kernels;
}

const std::vector<std::vector<float>> &
ConvNeuralNetwork::getKernelBiases() const {
	return this->kernelBiases;
}

// =============================================================================
// Serialisation
// =============================================================================

nlohmann::json
ConvNeuralNetwork::serialize(const ConvNeuralNetwork &cnn,
							 const std::string		 &filename) noexcept {
	nlohmann::json j;
	j["inputWidth"] = cnn.getInputWidth();
	j["inputHeight"] = cnn.getInputHeight();
	j["numFilters"] = cnn.getNumFilters();
	j["numConvLayers"] = cnn.getNumConvLayers();
	j["kernelSize"] = cnn.getKernelSize();
	j["convLearnRate"] = cnn.getConvLearnRate();

	const size_t nLayers = cnn.getNumConvLayers();
	const size_t nFilters = cnn.getNumFilters();
	const size_t kSz = cnn.getKernelSize();

	for (size_t l = 0; l < nLayers; ++l) {
		const size_t nIn = (l == 0) ? 1 : nFilters;
		for (size_t f = 0; f < nFilters; ++f) {
			j["kernelBiases"][l][f] = cnn.getKernelBiases()[l][f];
			for (size_t c = 0; c < nIn; ++c) {
				const DMatrix				   &k = cnn.getKernels()[l][f][c];
				std::vector<std::vector<float>> kd(kSz,
												   std::vector<float>(kSz));
				for (size_t r = 0; r < kSz; ++r)
					for (size_t col = 0; col < kSz; ++col)
						kd[r][col] = k(r, col);
				j["kernels"][l][f][c] = kd;
			}
		}
	}
	j["classifier"] = NeuralNetwork::serialize(cnn.getClassifier(), "");

	if (!filename.empty()) {
		std::ofstream file(filename);
		if (file.is_open())
			file << j.dump(4);
		else
			std::cerr << "[ConvNN] Error: cannot open '" << filename
					  << "' for writing\n";
	}
	return j;
}

ConvNeuralNetwork
ConvNeuralNetwork::deserialize(const std::string &filename) noexcept {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "[ConvNN] Error: cannot open '" << filename
				  << "' for reading\n";
		return ConvNeuralNetwork();
	}
	nlohmann::json j;
	file >> j;
	file.close();

	const size_t imgW = j["inputWidth"];
	const size_t imgH = j["inputHeight"];
	const size_t nFilters = j["numFilters"];
	const size_t nConvLayers =
		j.value("numConvLayers", j.value("filtersDepth", size_t(1)));
	const size_t kSz = j["kernelSize"];
	const float	 lr = j["convLearnRate"];

	// Build a shell (hiddenNodes / outputNodes / hiddenLayerLen = 0);
	// the classifier will be overwritten by setClassifier below.
	ConvNeuralNetwork cnn(imgW, imgH, nFilters, nConvLayers, kSz, 0, 0, 1);
	cnn.setConvLearnRate(lr);

	for (size_t l = 0; l < nConvLayers; ++l) {
		const size_t nIn = (l == 0) ? 1 : nFilters;
		for (size_t f = 0; f < nFilters; ++f) {
			cnn.kernelBiases[l][f] = j["kernelBiases"][l][f].get<float>();
			for (size_t c = 0; c < nIn; ++c) {
				const auto &kd = j["kernels"][l][f][c];
				DMatrix		k(kSz, kSz);
				for (size_t r = 0; r < kSz; ++r)
					for (size_t col = 0; col < kSz; ++col)
						k(r, col) = kd[r][col].get<float>();
				cnn.kernels[l][f][c] = std::move(k);
			}
		}
	}
	cnn.setClassifier(classifierFromJson(j["classifier"]));
	return cnn;
}

NeuralNetwork ConvNeuralNetwork::classifierFromJson(const nlohmann::json &j) {
	NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
					 j.at("numberOfHiddenNodes").get<size_t>(),
					 j.at("numberOfOutputNodes").get<size_t>(),
					 j.at("hiddenLayerLen").get<size_t>());
	nn.setLearnRate(j.at("learnRate").get<float>());
	for (size_t i = 0; i <= nn.getHiddenLayerLength(); ++i) {
		for (size_t r = 0; r < nn.getWeightAt(i).getRowLength(); ++r)
			for (size_t c = 0; c < nn.getWeightAt(i).getColLength(); ++c)
				nn.getWeightAt(i)(r, c) = j["weights"][i][r][c].get<float>();
		for (size_t r = 0; r < nn.getBiasAt(i).getRowLength(); ++r)
			for (size_t c = 0; c < nn.getBiasAt(i).getColLength(); ++c)
				nn.getBiasAt(i)(r, c) = j["bias"][i][r][c].get<float>();
	}
	return nn;
}
