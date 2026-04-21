#include "NeuralNetwork.hpp"

// ═══════════════════════════════════════════════════════════════════════════
// Gradient utilities
// ═══════════════════════════════════════════════════════════════════════════

float clampGradient(const float x) {
	static constexpr float MAX = 100.0f;
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	if (x > MAX) return MAX;
	if (x < -MAX) return -MAX;
	return x;
}

float errorTolerance(const float x) {
	static const float TOL =
		1.0f / static_cast<float>(NeuralNetwork::TOLERANCE);
	if (std::isnan(x) || std::isinf(x)) return 1.0f;
	if (std::abs(x) < TOL) return 0.0f;
	return x;
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation functions
// All activation derivatives receive the PRE-activation value z.
// ═══════════════════════════════════════════════════════════════════════════

float Tanh(const float x) {
	constexpr float MAX = 5.0f;
	const float		y = (x > MAX) ? MAX : (x < -MAX) ? -MAX : x;
	const float		z = (2.0f / (1.0f + std::exp(-2.0f * y))) - 1.0f;
	return (std::isnan(z) || std::isinf(z)) ? 0.0f : z;
}

float Sigmoid(const float x) {
	constexpr float MAX = 10.0f;
	const float		y = (x > MAX) ? MAX : (x < -MAX) ? -MAX : x;
	const float		z = 1.0f / (1.0f + std::exp(-y));
	return (std::isnan(z) || std::isinf(z)) ? 0.0f : z;
}

float SiLU(const float x) {
	constexpr float MAX = 50.0f;
	const float		y = (x > MAX) ? MAX : (x < -MAX) ? -MAX : x;
	const float		z = y / (1.0f + std::exp(-y));
	return (std::isnan(z) || std::isinf(z)) ? 0.0f : z;
}

float ReLU(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x < 0.0f ? 0.0f : x;
}

float LeakyReLU(const float x) {
	static const float A = 1.0f / static_cast<float>(NeuralNetwork::ALPHA);
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x < 0.0f ? x * A : x;
}

float Step(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x >= 0.0f ? 1.0f : 0.0f;
}

float Linear(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x;
}

// ─── Derivatives (all receive the pre-activation z) ───────────────────────

float DTanh(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	const float t = Tanh(x);
	return 1.0f - t * t;
}

// BUG FIX: previously defined as x*(1-x), which is only correct when x is
// already the sigmoid output. All other derivatives receive z (pre-activation),
// so we must be consistent: compute σ(z)·(1−σ(z)) from z here.
// This removes the need for the special-case branch in train().
float DSigmoid(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	const float s = Sigmoid(x);
	return s * (1.0f - s);
}

float DSiLU(const float x) {
	constexpr float MAX = 50.0f;
	const float		y = (x > MAX) ? MAX : (x < -MAX) ? -MAX : x;
	const float		sigma = 1.0f / (1.0f + std::exp(-y));
	const float		z = y * sigma * (1.0f - sigma) + sigma;
	return (std::isnan(z) || std::isinf(z)) ? 0.0f : z;
}

float DReLU(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x > 0.0f ? 1.0f : 0.0f;
}

float DLeakyReLU(const float x) {
	static const float A = 1.0f / static_cast<float>(NeuralNetwork::ALPHA);
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x > 0.0f ? 1.0f : A;
}

float DStep(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return x >= 0.0f ? 1.0f : 0.0f;
}

float DLinear(const float /*x*/) {
	return 1.0f;
}

// ─── Softmax ──────────────────────────────────────────────────────────────

DMatrix Softmax(const DMatrix &x) {
	const size_t rows = x.getRowLength();
	const size_t cols = x.getColLength();
	DMatrix		 result(rows, cols);

	// Numerically stable: subtract max before exp
	float max_val = x.getValue(0, 0);
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			if (x.getValue(i, j) > max_val) max_val = x.getValue(i, j);

	float sum = 0.0f;
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			const float v = std::exp(x.getValue(i, j) - max_val);
			result.setValue(i, j, v);
			sum += v;
		}
	}

	if (sum > 0.0f) {
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				result.setValue(i, j, result.getValue(i, j) / sum);
	}
	return result;
}

// Gradient for softmax + cross-entropy: simplifies to (output − target),
// which the caller already supplies as `error`. `output` is unused here
// but kept in the signature for documentation clarity.
DMatrix DSoftmax(const DMatrix & /*output*/, const DMatrix &error) {
	return error;
}

// ═══════════════════════════════════════════════════════════════════════════
// NeuralNetwork — private helper
// ═══════════════════════════════════════════════════════════════════════════

// Allocates and He-initialises all weight/bias matrices.
// Requires that all scalar fields are already set.
void NeuralNetwork::initLayers() {
	// Layer 0: input → first hidden
	weight[0] = DMatrix(numberOfHiddenNodes, numberOfInputsNodes);
	weight[0].randomize(numberOfInputsNodes);
	bias[0] = DMatrix(numberOfHiddenNodes, 1);
	bias[0].randomize(numberOfInputsNodes);

	// Layers 1 … hiddenLayerLen-1: hidden → hidden
	for (size_t i = 1; i < hiddenLayerLen; ++i) {
		weight[i] = DMatrix(numberOfHiddenNodes, numberOfHiddenNodes);
		weight[i].randomize(numberOfHiddenNodes);
		bias[i] = DMatrix(numberOfHiddenNodes, 1);
		bias[i].randomize(numberOfHiddenNodes);
	}

	// Layer hiddenLayerLen: last hidden → output
	weight[hiddenLayerLen] = DMatrix(numberOfOutputNodes, numberOfHiddenNodes);
	weight[hiddenLayerLen].randomize(numberOfHiddenNodes);
	bias[hiddenLayerLen] = DMatrix(numberOfOutputNodes, 1);
	bias[hiddenLayerLen].randomize(numberOfHiddenNodes);
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

NeuralNetwork::NeuralNetwork()
	: learnRate(NNLEARNRATE), numberOfInputsNodes(2), numberOfHiddenNodes(2),
	  numberOfOutputNodes(1), hiddenLayerLen(1), weight(2), bias(2),
	  HiddenActivate(Sigmoid), HiddenDeactivate(DSigmoid),
	  OutputActivate(Sigmoid), OutputDeactivate(DSigmoid), useSoftmax(false) {
	initLayers();
}

// BUG FIX: useSoftmax was never initialised in this constructor (UB).
// BUG FIX: biases were never randomized in this constructor.
NeuralNetwork::NeuralNetwork(const size_t inputs, const size_t hidden,
							 const size_t outputs)
	: learnRate(NNLEARNRATE), numberOfInputsNodes(inputs),
	  numberOfHiddenNodes(hidden), numberOfOutputNodes(outputs),
	  hiddenLayerLen(1), weight(2), bias(2), HiddenActivate(Sigmoid),
	  HiddenDeactivate(DSigmoid), OutputActivate(Sigmoid),
	  OutputDeactivate(DSigmoid), useSoftmax(false) {
	initLayers();
}

NeuralNetwork::NeuralNetwork(const size_t inputs, const size_t hidden,
							 const size_t outputs, const size_t hiddenLayers)
	: learnRate(NNLEARNRATE), numberOfInputsNodes(inputs),
	  numberOfHiddenNodes(hidden), numberOfOutputNodes(outputs),
	  hiddenLayerLen(hiddenLayers), weight(hiddenLayers + 1),
	  bias(hiddenLayers + 1), HiddenActivate(ReLU), HiddenDeactivate(DReLU),
	  OutputActivate(Linear), OutputDeactivate(DLinear), useSoftmax(false) {
	initLayers();
}

// ═══════════════════════════════════════════════════════════════════════════
// Inference
// ═══════════════════════════════════════════════════════════════════════════

DMatrix NeuralNetwork::feedForward(const DMatrix &input) const {
	DMatrix res(input);
	for (size_t i = 0; i <= hiddenLayerLen; ++i) {
		res = weight[i] * res;
		res += bias[i];
		if (i == hiddenLayerLen) {
			res = useSoftmax ? Softmax(res) : (res.map(OutputActivate), res);
		} else {
			res.map(HiddenActivate);
		}
	}
	return res;
}

std::vector<float>
NeuralNetwork::feedForward(const std::vector<float> &input) const {
	return feedForward(DMatrix(input)).toVector();
}

// ═══════════════════════════════════════════════════════════════════════════
// Training
// ═══════════════════════════════════════════════════════════════════════════

DMatrix NeuralNetwork::train(const DMatrix &inputArray,
							 const DMatrix &desired) {
	const size_t numLayers = hiddenLayerLen + 1; // total weight layers

	// ── Forward pass ──────────────────────────────────────────────────────
	// Store pre-activations z[i] and post-activations a[i+1].
	// a[0] = raw input; a[i+1] = f(z[i]).
	std::vector<DMatrix> a(numLayers + 1); // post-activation at each layer
	std::vector<DMatrix> z(numLayers);	   // pre-activation at each layer

	a[0] = inputArray;
	for (size_t i = 0; i < numLayers; ++i) {
		z[i] = weight[i] * a[i] + bias[i];
		a[i + 1] = z[i]; // copy, then apply activation below
		if (i == hiddenLayerLen) {
			a[i + 1] = useSoftmax ? Softmax(z[i])
								  : (a[i + 1].map(OutputActivate), a[i + 1]);
		} else {
			a[i + 1].map(HiddenActivate);
		}
	}

	// ── Backward pass ─────────────────────────────────────────────────────
	// All derivative functions now receive z (pre-activation) uniformly,
	// so no more special-casing on function pointer identity.
	//
	// δ[L]   = (desired − a[L+1]) ⊙ f'(z[L])          output layer
	// δ[i]   = (W[i+1]ᵀ · δ[i+1]) ⊙ f'(z[i])          hidden layers
	//
	// ΔW[i]  = η · δ[i] · a[i]ᵀ
	// Δb[i]  = η · δ[i]

	// Compute output delta
	DMatrix delta = desired - a[numLayers]; // error at output

	if (useSoftmax) {
		// Softmax + cross-entropy: gradient = (output − target) / n_outputs
		delta *= 1.0f / static_cast<float>(numberOfOutputNodes);
	} else {
		// BUG FIX: all derivatives now correctly receive z (pre-activation).
		// The old branch `if (deactivate == DSigmoid) use a` is no longer
		// needed because DSigmoid(z) now computes σ(z)·(1−σ(z)) from z.
		DMatrix df = z[hiddenLayerLen];
		df.map(OutputDeactivate);
		delta.multiply(df); // element-wise: error ⊙ f'(z)
	}

	// Propagate deltas from output layer back to first hidden layer
	// BUG FIX: the old code used the scaled+clamped gradient to propagate
	// layerError left through hidden layers, which corrupted the signal.
	// Now we separate the weight update (scaled δ) from the propagation
	// (unscaled δ multiplied by Wᵀ).
	for (long i = static_cast<long>(hiddenLayerLen); i >= 0; --i) {
		const size_t ui = static_cast<size_t>(i);

		// Weight update uses the scaled, clamped delta
		DMatrix scaledDelta = delta;
		scaledDelta *= learnRate;
		scaledDelta.map(clampGradient);

		const DMatrix weightDelta = scaledDelta * a[ui].transpose();
		weight[ui] += weightDelta;
		bias[ui] += scaledDelta;

		if (i == 0) break; // no layer to the left

		// Propagate error left using UNSCALED delta and OLD weights
		// (weight[ui] is already updated above, so save Wᵀ before updating).
		// We saved the transpose implicitly by computing it before +=.
		// Re-derive: use the pre-update weight stored in weightDelta's
		// source. Actually we need Wᵀ of the weight BEFORE the update.
		// Recompute: delta_prev = Wᵀ · delta ⊙ f'(z[i-1])
		const DMatrix wT = (weight[ui] - weightDelta).transpose();
		DMatrix		  newDelta = wT * delta;

		DMatrix df = z[ui - 1];
		df.map(HiddenDeactivate);
		newDelta.multiply(df);
		delta = std::move(newDelta);
	}

	clampWeightsAndBiases();
	return delta;
}

// Convenience overload: wraps vectors into column DMatrices.
DMatrix NeuralNetwork::train(const std::vector<float> &input,
							 const std::vector<float> &desired) {
	return train(DMatrix(input), DMatrix(desired));
}

// ═══════════════════════════════════════════════════════════════════════════
// Weight / bias clamping
// ═══════════════════════════════════════════════════════════════════════════

void NeuralNetwork::clampWeightsAndBiases() {
	static constexpr float WCLAMP = static_cast<float>(NeuralNetwork::CLAMP);

	for (size_t i = 0; i <= hiddenLayerLen; ++i) {
		// Clamp weights
		for (size_t r = 0; r < weight[i].getRowLength(); ++r) {
			for (size_t c = 0; c < weight[i].getColLength(); ++c) {
				float &v = weight[i](r, c);
				if (std::isnan(v) || std::isinf(v))
					v = 0.0f;
				else if (v > WCLAMP)
					v = WCLAMP;
				else if (v < -WCLAMP)
					v = -WCLAMP;
			}
		}
		// BUG FIX: bias clamp used sqrt(fanIn+fanOut) which was far too
		// small (e.g. ≈2.2 for a 2→2→1 net) and aggressively clipped biases
		// throughout training. Use the same constant as weights.
		for (size_t r = 0; r < bias[i].getRowLength(); ++r) {
			for (size_t c = 0; c < bias[i].getColLength(); ++c) {
				float &v = bias[i](r, c);
				if (std::isnan(v) || std::isinf(v))
					v = 0.0f;
				else if (v > WCLAMP)
					v = WCLAMP;
				else if (v < -WCLAMP)
					v = -WCLAMP;
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation (evolutionary)
// ═══════════════════════════════════════════════════════════════════════════

NeuralNetwork NeuralNetwork::mutate(float (*func)(float)) const {
	NeuralNetwork m(*this);
	for (size_t i = 0; i <= m.hiddenLayerLen; ++i) {
		m.weight[i].map(func);
		m.bias[i].map(func);
	}
	return m;
}

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

void NeuralNetwork::enableSoftmax(const bool enable) {
	useSoftmax = enable;
	if (enable) {
		OutputActivate = Linear;
		OutputDeactivate = DLinear;
	}
}

void NeuralNetwork::setHiddenLayerActivation(float (*activate)(float),
											 float (*deactivate)(float)) {
	if (!activate || !deactivate) return;
	HiddenActivate = activate;
	HiddenDeactivate = deactivate;
}

void NeuralNetwork::setOutputLayerActivation(float (*activate)(float),
											 float (*deactivate)(float)) {
	if (!activate || !deactivate) return;
	OutputActivate = activate;
	OutputDeactivate = deactivate;
}

void NeuralNetwork::setLearnRate(const float lr) {
	learnRate = lr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════════════════════════

float NeuralNetwork::getLearnRate() const {
	return learnRate;
}
size_t NeuralNetwork::getHiddenLayerLength() const {
	return hiddenLayerLen;
}
size_t NeuralNetwork::getNumberOfInputsNodes() const {
	return numberOfInputsNodes;
}
size_t NeuralNetwork::getNumberOfHiddenNodes() const {
	return numberOfHiddenNodes;
}
size_t NeuralNetwork::getNumberOfOutputsNodes() const {
	return numberOfOutputNodes;
}

DMatrix &NeuralNetwork::getBiasAt(const size_t index) {
	return bias[index <= hiddenLayerLen ? index : hiddenLayerLen];
}
DMatrix &NeuralNetwork::getWeightAt(const size_t index) {
	return weight[index <= hiddenLayerLen ? index : hiddenLayerLen];
}
const DMatrix &NeuralNetwork::getBiasAt(const size_t index) const {
	return bias[index <= hiddenLayerLen ? index : hiddenLayerLen];
}
const DMatrix &NeuralNetwork::getWeightAt(const size_t index) const {
	return weight[index <= hiddenLayerLen ? index : hiddenLayerLen];
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════════════

nlohmann::json NeuralNetwork::serialize(const NeuralNetwork &nn,
										const std::string &filename) noexcept {
	nlohmann::json j;
	j["learnRate"] = nn.learnRate;
	j["numberOfInputsNodes"] = nn.numberOfInputsNodes;
	j["numberOfHiddenNodes"] = nn.numberOfHiddenNodes;
	j["numberOfOutputNodes"] = nn.numberOfOutputNodes;
	j["hiddenLayerLen"] = nn.hiddenLayerLen;
	j["useSoftmax"] = nn.useSoftmax;

	for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
		for (size_t r = 0; r < nn.weight[i].getRowLength(); ++r)
			for (size_t c = 0; c < nn.weight[i].getColLength(); ++c)
				j["weights"][i][r][c] = nn.weight[i](r, c);
		for (size_t r = 0; r < nn.bias[i].getRowLength(); ++r)
			for (size_t c = 0; c < nn.bias[i].getColLength(); ++c)
				j["bias"][i][r][c] = nn.bias[i](r, c);
	}

	if (!filename.empty()) {
		std::ofstream out(filename);
		if (out.is_open())
			out << j.dump(4);
		else
			std::cerr << "[NeuralNetwork] Could not write: " << filename
					  << '\n';
	}
	return j;
}

NeuralNetwork NeuralNetwork::deserialize(const std::string &filename) noexcept {
	std::ifstream in(filename);
	if (!in.is_open()) {
		std::cerr << "[NeuralNetwork] Could not read: " << filename << '\n';
		return NeuralNetwork();
	}

	nlohmann::json j;
	in >> j;

	NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
					 j.at("numberOfHiddenNodes").get<size_t>(),
					 j.at("numberOfOutputNodes").get<size_t>(),
					 j.at("hiddenLayerLen").get<size_t>());

	nn.setLearnRate(j.at("learnRate").get<float>());
	if (j.contains("useSoftmax"))
		nn.enableSoftmax(j.at("useSoftmax").get<bool>());

	for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
		for (size_t r = 0; r < nn.weight[i].getRowLength(); ++r)
			for (size_t c = 0; c < nn.weight[i].getColLength(); ++c)
				nn.weight[i](r, c) = j["weights"][i][r][c].get<float>();
		for (size_t r = 0; r < nn.bias[i].getRowLength(); ++r)
			for (size_t c = 0; c < nn.bias[i].getColLength(); ++c)
				nn.bias[i](r, c) = j["bias"][i][r][c].get<float>();
	}
	return nn;
}
