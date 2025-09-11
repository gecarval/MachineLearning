#include "NeuralNetwork.hpp"

static float clampGradient(const float x) {
	static const float max = NeuralNetwork::CLAMP;
	float			   r = x;
	if (std::abs(r) > max) r = r > 0 ? max : -max;
	return (r);
}

static float errorTolerance(const float x) {
	static const float tol = 1.0f / NeuralNetwork::TOLERANCE;
	if (std::abs(x) < tol) return (0);
	return (x);
}

static float sigmoid(const float x) {
	return (1.0f / (1.0f + std::exp(-x)));
}

static float relu(const float x) {
	static const float a = 1.0f / NeuralNetwork::ALPHA;
	if (x < 0) return (x * a);
	return (x);
}

static float dSigmoid(const float y) {
	return (y * (1.0f - y));
}

static float dRelu(const float y) {
	static const float a = 1.0f / NeuralNetwork::ALPHA;
	return (y >= 0 ? 1.0f : a);
}

NeuralNetwork::NeuralNetwork()
	: learnRate(NNLEARNRATE), numberOfInputsNodes(2), numberOfHiddenNodes(2),
	  numberOfOutputNodes(1), hiddenLayerLen(1), weight(new DMatrix[2]),
	  bias(new DMatrix[2]) {
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
	  weight(new DMatrix[2]), bias(new DMatrix[2]) {
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
	  bias(new DMatrix[hiddenLayerLength + 1]) {
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

NeuralNetwork::NeuralNetwork(const NeuralNetwork &other) {
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
		delete[] this->weight;
		this->weight = new DMatrix[this->hiddenLayerLen + 1];
		for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
			this->weight[i] = other.weight[i];
		delete[] this->bias;
		this->bias = new DMatrix[this->hiddenLayerLen + 1];
		for (size_t i = 0; i < this->hiddenLayerLen + 1; i++)
			this->bias[i] = other.bias[i];
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
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		res = this->weight[i] * res;
		res += this->bias[i];
		if (i == this->hiddenLayerLen)
			res.map(sigmoid);
		else
			res.map(relu);
	}
	return (res);
}

void NeuralNetwork::train(const DMatrix &inputArray, const DMatrix &desired) {
	std::vector<DMatrix> outputs(this->hiddenLayerLen + 1);
	DMatrix				 res(inputArray);
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		res = this->weight[i] * res;
		res += this->bias[i];
		if (i == this->hiddenLayerLen) {
			res.map(sigmoid);
			outputs[i] = res;
		} else {
			res.map(relu);
			outputs[i] = res;
		}
	}
	DMatrix layerError(desired - outputs[this->hiddenLayerLen]);
	for (long i = this->hiddenLayerLen; i >= 0; i--) {
		if ((size_t)i != this->hiddenLayerLen)
			layerError = this->weight[i + 1].transpose() * layerError;
		layerError.map(errorTolerance);
		DMatrix gradient(outputs[i]);
		if ((size_t)i != this->hiddenLayerLen)
			gradient.map(dRelu);
		else
			gradient.map(dSigmoid);
		gradient.multiply(layerError);
		gradient *= this->learnRate;
		gradient.map(clampGradient);
		const DMatrix &transposed =
			i == 0 ? inputArray.transpose() : outputs[i].transpose();
		const DMatrix weightDelta(gradient * transposed);
		this->weight[i] += weightDelta;
		this->bias[i] += gradient;
	}
}

const DMatrix &NeuralNetwork::getBiasAt(const size_t index) const {
	if (index > this->hiddenLayerLen) return (this->bias[this->hiddenLayerLen]);
	return (this->bias[index]);
}

const DMatrix &NeuralNetwork::getWeigthAt(const size_t index) const {
	if (index > this->hiddenLayerLen)
		return (this->weight[this->hiddenLayerLen]);
	return (this->weight[index]);
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
