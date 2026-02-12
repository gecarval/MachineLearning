#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "../includes/json.hpp"
#include "./DMatrix.hpp"
#include "fstream"
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef NNLEARNRATE
#define NNLEARNRATE 0.001f
#endif

// Gradient Descent
float clampGradient(const float x);
float errorTolerance(const float x);

// Activation Functions
float Tanh(const float x);
float Sigmoid(const float x);
float SiLU(const float x);
float ReLU(const float x);
float LeakyReLU(const float x);
float Step(const float x);
float Linear(const float x);

// Deactivation Functions
float DTanh(const float x);
float DSigmoid(const float x);
float DSiLU(const float x);
float DReLU(const float x);
float DLeakyReLU(const float x);
float DStep(const float x);
float DLinear(const float x);

// Softmax functions
DMatrix Softmax(const DMatrix &x);
DMatrix DSoftmax(const DMatrix &output, const DMatrix &error);

class NeuralNetwork {
  protected:
	float				 learnRate;
	size_t				 numberOfInputsNodes;
	size_t				 numberOfHiddenNodes;
	size_t				 numberOfOutputNodes;
	size_t				 hiddenLayerLen;
	std::vector<DMatrix> weight;
	std::vector<DMatrix> bias;
	float (*HiddenActivate)(float);
	float (*HiddenDeactivate)(float);
	float (*OutputActivate)(float);
	float (*OutputDeactivate)(float);
	bool useSoftmax;

  public:
	static const int CLAMP = 10000;
	static const int TOLERANCE = 10000;
	static const int ALPHA = 100;

	virtual ~NeuralNetwork();
	NeuralNetwork();
	NeuralNetwork(const size_t numberOfInputsNodes,
				  const size_t numberOfHiddenNodes,
				  const size_t numberOfOutputNodes);
	NeuralNetwork(const size_t numberOfInputsNodes,
				  const size_t numberOfHiddenNodes,
				  const size_t numberOfOutputNodes,
				  const size_t hiddenLayerLength);
	NeuralNetwork(const NeuralNetwork &other);
	NeuralNetwork &operator=(const NeuralNetwork &other);

	DMatrix feedForward(const DMatrix &input) const;
	virtual std::vector<float>
	feedForward(const std::vector<float> &input) const;

	virtual DMatrix train(const DMatrix &inputArray, const DMatrix &desired);
	void			clampWeightsAndBiases();

	void		   setHiddenLayerActivation(float (*Activate)(float),
											float (*Deactivate)(float));
	void		   setOutputLayerActivation(float (*Activate)(float),
											float (*Deactivate)(float));
	void		   enableSoftmax(bool enable);
	void		   setLearnRate(const float newLearnRate);
	float		   getLearnRate(void) const;
	size_t		   getHiddenLayerLength(void) const;
	size_t		   getNumberOfInputsNodes(void) const;
	size_t		   getNumberOfHiddenNodes(void) const;
	size_t		   getNumberOfOutputsNodes(void) const;
	const DMatrix &getBiasAt(const size_t index) const;
	const DMatrix &getWeightAt(const size_t index) const;

	virtual NeuralNetwork mutate(float (*func)(float)) const;

	// Serialize NeuralNetwork to a JSON file
	static void serialize(const NeuralNetwork &nn,
						  const std::string	  &filename) {
		std::ofstream out(filename);
		if (!out.is_open()) {
			throw std::runtime_error("Unable to open file for serialization: " +
									 filename);
		}
		nlohmann::json j;
		j["learnRate"] = nn.learnRate;
		j["numberOfInputsNodes"] = nn.numberOfInputsNodes;
		j["numberOfHiddenNodes"] = nn.numberOfHiddenNodes;
		j["numberOfOutputNodes"] = nn.numberOfOutputNodes;
		j["hiddenLayerLen"] = nn.hiddenLayerLen;
		// Serialize Weights
		for (const auto &weightMatrix : nn.weight) {
			// DMatrix::toVector() returns the underlying std::vector<float>
			j["weights"].push_back(weightMatrix.toVector());
		}
		// Serialize Biases
		for (const auto &biasMatrix : nn.bias) {
			j["biases"].push_back(biasMatrix.toVector());
		}
		out << j.dump(4); // Use 4 spaces for indentation
		out.close();
	}

	// Deserialize NeuralNetwork from a JSON file
	static NeuralNetwork deserialize(const std::string &filename) {
		std::ifstream in(filename);
		if (!in.is_open()) {
			throw std::runtime_error(
				"Unable to open file for deserialization: " + filename);
		}
		nlohmann::json j;
		in >> j;
		// Create the NN instance using the loaded parameters
		NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
						 j.at("numberOfHiddenNodes").get<size_t>(),
						 j.at("numberOfOutputNodes").get<size_t>(),
						 j.at("hiddenLayerLen").get<size_t>());
		nn.setLearnRate(j.at("learnRate").get<float>());
		// Load Weights
		for (size_t i = 0; i < nn.hiddenLayerLen + 1; ++i) {
			std::vector<float> w_data =
				j.at("weights").at(i).get<std::vector<float>>();
			// We must reshape the flat vector back to a Matrix (rows, cols)
			size_t r = nn.weight[i].getRowLength();
			size_t c = nn.weight[i].getColLength();
			nn.weight[i] = DMatrix(w_data);
			// Manual Reshape: Since DMatrix(vector) creates a column vector,
			// we manually fix dimensions
			nn.weight[i] = DMatrix(r, c);
			for (size_t idx = 0; idx < w_data.size(); ++idx) {
				nn.weight[i](idx / c, idx % c) = w_data[idx];
			}
		}
		// Load Biases
		for (size_t i = 0; i < nn.hiddenLayerLen + 1; ++i) {
			std::vector<float> b_data =
				j.at("biases").at(i).get<std::vector<float>>();
			nn.bias[i] = DMatrix(b_data);
		}
		return (nn);
	}
};

#endif
