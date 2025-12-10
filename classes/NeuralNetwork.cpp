#include "NeuralNetwork.hpp"

float clampGradient(const float x) {
	static const float max = 100.0f;
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (std::abs(x) < max ? x : x > 0.0f ? max : -max);
}

float errorTolerance(const float x) {
	static const float tol = 1.0f / NeuralNetwork::TOLERANCE;
	if (std::isnan(x) || std::isinf(x)) return (1.0f);
	if (std::abs(x) < tol) return (0.0f);
	return (x);
}

float Tanh(const float x) {
	const float max = 5.0f;
	const float y = std::abs(x) < max ? x : x > 0.0f ? max : -max;
	const float z = (2.0f / (1.0f + std::exp(-2 * y))) - 1.0f;
	if (std::isnan(z) || std::isinf(z)) return (0.0f);
	return (z);
}

float Sigmoid(const float x) {
	const float max = 10.0f;
	const float y = std::abs(x) < max ? x : x > 0.0f ? max : -max;
	const float z = 1.0f / (1.0f + std::exp(-y));
	if (std::isnan(z) || std::isinf(z)) return (0.0f);
	return (z);
}

float SiLU(const float x) {
	const float max = 50.0f;
	const float y = std::abs(x) < max ? x : x > 0.0f ? max : -max;
	const float z = y / (1.0f + std::exp(-y));
	if (std::isnan(z) || std::isinf(z)) return 0.0f;
	return z;
}

float ReLU(const float x) {
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (x < 0.0f ? 0.0f : x);
}

float LeakyReLU(const float x) {
	static const float a = 1.0f / NeuralNetwork::ALPHA;
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (x < 0.0f ? x * a : x);
}

float Step(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return (x >= 0.0f) ? 1.0f : 0.0f;
}

float DTanh(const float x) {
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	const float y = Tanh(x);
	return (1.0f - (y * y));
}

float DSigmoid(const float x) {
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (x * (1.0f - x));
}

float DSiLU(const float x) {
	const float max = 50.0f;
	const float y = std::abs(x) < max ? x : x > 0.0f ? max : -max;
	const float sigma = 1.0f / (1.0f + std::exp(-y));
	const float z = y * sigma * (1.0f - sigma) + sigma;
	if (std::isnan(z) || std::isinf(z)) return 0.0f;
	return (z);
}

float DReLU(const float x) {
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (x > 0.0f ? 1.0f : 0.0f);
}

float DLeakyReLU(const float x) {
	static const float a = 1.0f / NeuralNetwork::ALPHA;
	if (std::isnan(x) || std::isinf(x)) return (0.0f);
	return (x > 0.0f ? 1.0f : a);
}

float DStep(const float x) {
	if (std::isnan(x) || std::isinf(x)) return 0.0f;
	return (x >= 0.0f) ? 1.0f : 0.0f;
}

NeuralNetwork::NeuralNetwork()
	: learnRate(NNLEARNRATE), numberOfInputsNodes(2), numberOfHiddenNodes(2),
	  numberOfOutputNodes(1), hiddenLayerLen(1), weight(new DMatrix[2]),
	  bias(new DMatrix[2]), HiddenActivate(Sigmoid), HiddenDeactivate(DSigmoid),
	  OutputActivate(Sigmoid), OutputDeactivate(DSigmoid) {
	this->weight[0] =
		DMatrix(this->numberOfHiddenNodes, this->numberOfInputsNodes);
	this->weight[1] =
		DMatrix(this->numberOfOutputNodes, this->numberOfHiddenNodes);
	this->bias[0] = DMatrix(this->numberOfHiddenNodes, 1);
	this->bias[1] = DMatrix(this->numberOfOutputNodes, 1);
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
		this->weight[i].randomize();
}

NeuralNetwork::NeuralNetwork(const size_t numberOfInputsNodes,
							 const size_t numberOfHiddenNodes,
							 const size_t numberOfOutputNodes)
	: learnRate(NNLEARNRATE), numberOfInputsNodes(numberOfInputsNodes),
	  numberOfHiddenNodes(numberOfHiddenNodes),
	  numberOfOutputNodes(numberOfOutputNodes), hiddenLayerLen(1),
	  weight(new DMatrix[2]), bias(new DMatrix[2]), HiddenActivate(Sigmoid),
	  HiddenDeactivate(DSigmoid), OutputActivate(Sigmoid),
	  OutputDeactivate(DSigmoid) {
	this->weight[0] =
		DMatrix(this->numberOfHiddenNodes, this->numberOfInputsNodes);
	this->weight[1] =
		DMatrix(this->numberOfOutputNodes, this->numberOfHiddenNodes);
	this->bias[0] = DMatrix(this->numberOfHiddenNodes, 1);
	this->bias[1] = DMatrix(this->numberOfOutputNodes, 1);
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
		this->weight[i].randomize();
}

NeuralNetwork::NeuralNetwork(const size_t numberOfInputsNodes,
							 const size_t numberOfHiddenNodes,
							 const size_t numberOfOutputNodes,
							 const size_t hiddenLayerLength)
	: learnRate(NNLEARNRATE), numberOfInputsNodes(numberOfInputsNodes),
	  numberOfHiddenNodes(numberOfHiddenNodes),
	  numberOfOutputNodes(numberOfOutputNodes),
	  hiddenLayerLen(hiddenLayerLength),
	  weight(new DMatrix[hiddenLayerLength + 1]),
	  bias(new DMatrix[hiddenLayerLength + 1]), HiddenActivate(Sigmoid),
	  HiddenDeactivate(DSigmoid), OutputActivate(Sigmoid),
	  OutputDeactivate(DSigmoid) {
	this->weight[0] =
		DMatrix(this->numberOfHiddenNodes, this->numberOfInputsNodes);
	this->weight[0].randomize(this->numberOfInputsNodes);
	this->bias[0] = DMatrix(this->numberOfHiddenNodes, 1);
	this->bias[0].randomize(this->numberOfInputsNodes);
	for (size_t i = 1; i < this->hiddenLayerLen; i++) {
		this->weight[i] =
			DMatrix(this->numberOfHiddenNodes, this->numberOfHiddenNodes);
		this->weight[i].randomize(this->numberOfHiddenNodes);
		this->bias[i] = DMatrix(this->numberOfHiddenNodes, 1);
		this->bias[i].randomize(this->numberOfHiddenNodes);
	}
	this->weight[this->hiddenLayerLen] =
		DMatrix(this->numberOfOutputNodes, this->numberOfHiddenNodes);
	this->weight[this->hiddenLayerLen].randomize(this->numberOfOutputNodes);
	this->bias[this->hiddenLayerLen] = DMatrix(this->numberOfOutputNodes, 1);
	this->bias[this->hiddenLayerLen].randomize(this->numberOfOutputNodes);
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork &other)
	: weight(NULL), bias(NULL) {
	*this = other;
}

NeuralNetwork::~NeuralNetwork() {
	delete[] this->weight;
	delete[] this->bias;
}

NeuralNetwork &NeuralNetwork::operator=(const NeuralNetwork &other) {
	if (this != &other) {
		this->learnRate = other.learnRate;
		this->numberOfInputsNodes = other.numberOfInputsNodes;
		this->numberOfHiddenNodes = other.numberOfHiddenNodes;
		this->numberOfOutputNodes = other.numberOfOutputNodes;
		this->hiddenLayerLen = other.hiddenLayerLen;
		if (this->weight != NULL) delete[] this->weight;
		this->weight = new DMatrix[this->hiddenLayerLen + 1];
		for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
			this->weight[i] = other.weight[i];
		if (this->bias != NULL) delete[] this->bias;
		this->bias = new DMatrix[this->hiddenLayerLen + 1];
		for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
			this->bias[i] = other.bias[i];
		this->HiddenActivate = other.HiddenActivate;
		this->HiddenDeactivate = other.HiddenDeactivate;
		this->OutputActivate = other.OutputActivate;
		this->OutputDeactivate = other.OutputDeactivate;
	}
	return (*this);
}

std::vector<float>
NeuralNetwork::feedFoward(const std::vector<float> &input) const {
	DMatrix res(input);
	res = this->feedFoward(res);
	return (res.toVector());
}

DMatrix NeuralNetwork::feedFoward(const DMatrix &input) const {
	DMatrix res(input);
	for (size_t i = 0; i <= this->hiddenLayerLen; i++) {
		res = this->weight[i] * res;
		res += this->bias[i];
		if (i == this->hiddenLayerLen) {
			res.map(this->OutputActivate);
		} else {
			res.map(this->HiddenActivate);
		}
	}
	return (res);
}

void NeuralNetwork::train(const DMatrix &inputArray, const DMatrix &desired) {
	std::vector<DMatrix> outputs(this->hiddenLayerLen + 1);
	DMatrix				 res(inputArray);
	for (size_t i = 0; i <= this->hiddenLayerLen; i++) {
		res = this->weight[i] * res;
		res += this->bias[i];
		if (i == this->hiddenLayerLen) {
			res.map(this->OutputActivate);
		} else {
			res.map(this->HiddenActivate);
		}
		outputs[i] = res;
	}
	DMatrix layerError(desired - outputs[this->hiddenLayerLen]);
	for (size_t i = this->hiddenLayerLen; i <= this->hiddenLayerLen; i--) {
		DMatrix gradient = outputs[i];
		if (i == this->hiddenLayerLen) {
			gradient.map(this->OutputDeactivate);
		} else {
			gradient.map(this->HiddenDeactivate);
		}
		gradient.multiply(layerError);
		gradient.map(errorTolerance);
		gradient *= this->learnRate;
		gradient.map(clampGradient);
		const DMatrix &transposed =
			(i == 0) ? inputArray.transpose() : outputs[i - 1].transpose();
		const DMatrix weightDelta(gradient * transposed);
		this->weight[i] += weightDelta;
		this->bias[i] += gradient;
		if (i > 0) {
			layerError = this->weight[i].transpose() * layerError;
		}
	}
	this->clampWeightsAndBiases();
}

void NeuralNetwork::clampWeightsAndBiases() {
	for (size_t i = 0; i < hiddenLayerLen + 1; ++i) {
		const size_t fan_in =
			(i == 0) ? numberOfInputsNodes : numberOfHiddenNodes;
		const size_t fan_out =
			(i == hiddenLayerLen) ? numberOfOutputNodes : numberOfHiddenNodes;
		const float weight_clamp = NeuralNetwork::CLAMP;
		const float bias_clamp = fan_in + fan_out;
		// Clamp weights
		const size_t weight_rows = weight[i].getRowLength();
		const size_t weight_cols = weight[i].getColLength();
		for (size_t r = 0; r < weight_rows; ++r) {
			for (size_t c = 0; c < weight_cols; ++c) {
				float &val = weight[i](r, c);
				if (std::isnan(val) || std::isinf(val)) {
					val = 0;
				} else if (std::abs(val) > weight_clamp) {
					val = val < 0 ? -weight_clamp : weight_clamp;
				}
			}
		}
		const size_t bias_rows = bias[i].getRowLength();
		const size_t bias_cols = bias[i].getColLength();
		for (size_t r = 0; r < bias_rows; ++r) {
			for (size_t c = 0; c < bias_cols; ++c) {
				float &val = bias[i](r, c);
				if (std::isnan(val) || std::isinf(val)) {
					val = 0;
				} else if (std::abs(val) > bias_clamp) {
					val = val < 0 ? -bias_clamp : bias_clamp;
				}
			}
		}
	}
}

NeuralNetwork NeuralNetwork::mutate(float (*func)(float)) const {
	NeuralNetwork m(*this);
	for (size_t i = 0; i < m.hiddenLayerLen + 1; i++) {
		m.weight[i].map(func);
		m.bias[i].map(func);
	}
	return (m);
}

const DMatrix &NeuralNetwork::getBiasAt(const size_t index) const {
	if (index > this->hiddenLayerLen) {
		return (this->bias[this->hiddenLayerLen]);
	}
	return (this->bias[index]);
}

const DMatrix &NeuralNetwork::getWeigthAt(const size_t index) const {
	if (index > this->hiddenLayerLen) {
		return (this->weight[this->hiddenLayerLen]);
	}
	return (this->weight[index]);
}

void NeuralNetwork::setHiddenLayerActivation(float (*Activate)(float),
											 float (*Deactivate)(float)) {
	if (!Activate || !Deactivate) {
		return;
	}
	this->HiddenActivate = Activate;
	this->HiddenDeactivate = Deactivate;
}

void NeuralNetwork::setOutputLayerActivation(float (*Activate)(float),
											 float (*Deactivate)(float)) {
	if (!Activate || !Deactivate) {
		return;
	}
	this->OutputActivate = Activate;
	this->OutputDeactivate = Deactivate;
}

void NeuralNetwork::setLearnRate(const float newLearnRate) {
	this->learnRate = newLearnRate;
}

float NeuralNetwork::getLearnRate(void) const {
	return (this->learnRate);
}

size_t NeuralNetwork::getHiddenLayerLength(void) const {
	return (this->hiddenLayerLen);
}

size_t NeuralNetwork::getNumberOfInputsNodes(void) const {
	return (this->numberOfInputsNodes);
}

size_t NeuralNetwork::getNumberOfHiddenNodes(void) const {
	return (this->numberOfHiddenNodes);
}

size_t NeuralNetwork::getNumberOfOutputsNodes(void) const {
	return (this->numberOfOutputNodes);
}
