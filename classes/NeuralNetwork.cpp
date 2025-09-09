#include "NeuralNetwork.hpp"

static float sigmoid(const float x) {
	return (1 / (1 + std::exp(-x)));
}

static float relu(const float x) {
	if (x < 0) return (0);
	return (x);
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
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		this->weight[i].randomize();
		this->bias[i].randomize();
	}
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
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		this->weight[i].randomize();
		this->bias[i].randomize();
	}
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
	this->bias[0] = DMatrix(this->numberOfHiddenNodes, 1);
	for (size_t i = 1; i < this->hiddenLayerLen; i++) {
		this->weight[i] =
			DMatrix(this->numberOfHiddenNodes, this->numberOfHiddenNodes);
		this->bias[i] = DMatrix(this->numberOfHiddenNodes, 1);
	}
	this->weight[this->hiddenLayerLen] =
		DMatrix(this->numberOfOutputNodes, this->numberOfHiddenNodes);
	this->bias[this->hiddenLayerLen] = DMatrix(this->numberOfOutputNodes, 1);
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		this->weight[i].randomize();
		this->bias[i].randomize();
	}
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
	for (size_t i = 0; i < this->hiddenLayerLen + 1; i++) {
		res = this->weight[i] * res;
		res += this->bias[i];
		if (i == this->hiddenLayerLen)
			res.map(sigmoid);
		else
			res.map(relu);
	}
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
	const DMatrix  output(this->feedFoward(inputArray));
	const DMatrix &target = desired;
	DMatrix		   outputError((target - output));
	for (size_t i = this->hiddenLayerLen; i != 0; i--) {
		outputError = this->weight[i].transpose() * outputError;
	}
}

void NeuralNetwork::train(const std::vector<float> &inputArray,
						  const std::vector<float> &desired) {
	const DMatrix output(this->feedFoward(inputArray));
	const DMatrix target(desired);
	DMatrix		  outputError((target - output));
	for (size_t i = this->hiddenLayerLen; i != 0; i--) {
		outputError = this->weight[i].transpose() * outputError;
	}
}

void NeuralNetwork::setLearnRate(const float newLearnRate) {
	this->learnRate = newLearnRate;
}

float NeuralNetwork::getLearnRate(void) const {
	return (this->learnRate);
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
