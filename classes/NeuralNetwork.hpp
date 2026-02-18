#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "../includes/json.hpp"
#include "./DMatrix.hpp"
#include "fstream"

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
	DMatrix		  &getBiasAt(const size_t index);
	DMatrix		  &getWeightAt(const size_t index);
	const DMatrix &getBiasAt(const size_t index) const;
	const DMatrix &getWeightAt(const size_t index) const;

	virtual NeuralNetwork mutate(float (*func)(float)) const;

	// Serialize NeuralNetwork to a JSON file
	static nlohmann::json serialize(const NeuralNetwork &nn,
									const std::string	&filename) noexcept {
		nlohmann::json j;
		j["learnRate"] = nn.learnRate;
		j["numberOfInputsNodes"] = nn.numberOfInputsNodes;
		j["numberOfHiddenNodes"] = nn.numberOfHiddenNodes;
		j["numberOfOutputNodes"] = nn.numberOfOutputNodes;
		j["hiddenLayerLen"] = nn.hiddenLayerLen;
		// Serialize Weights
		for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
			for (size_t r = 0; r < nn.weight[i].getRowLength(); ++r) {
				for (size_t c = 0; c < nn.weight[i].getColLength(); ++c) {
					j["weights"][i][r][c] = nn.weight[i](r, c);
				}
			}
		}
		// Serialize Biases
		for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
			for (size_t r = 0; r < nn.bias[i].getRowLength(); ++r) {
				for (size_t c = 0; c < nn.bias[i].getColLength(); ++c) {
					j["bias"][i][r][c] = nn.bias[i](r, c);
				}
			}
		}
		// Dump JSON with 4 space indentation
		std::ofstream out(filename);
		if (out.is_open()) {
			out << j.dump(4);
			out.close();
		} else if (filename != "") {
			std::cerr << "Error: Could not open file for writing: " << filename
					  << std::endl;
		}
		return (j);
	}

	// Deserialize NeuralNetwork from a JSON file
	static NeuralNetwork deserialize(const std::string &filename) noexcept {
		nlohmann::json j;
		std::ifstream  in(filename);
		if (!in.is_open()) {
			std::cerr << "Error: Could not open file for reading: " << filename
					  << std::endl;
			return (NeuralNetwork());
		}
		in >> j;
		// Create the NN instance using the loaded parameters
		NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
						 j.at("numberOfHiddenNodes").get<size_t>(),
						 j.at("numberOfOutputNodes").get<size_t>(),
						 j.at("hiddenLayerLen").get<size_t>());
		nn.setLearnRate(j.at("learnRate").get<float>());
		// Load Weights
		for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
			for (size_t r = 0; r < nn.weight[i].getRowLength(); ++r) {
				for (size_t c = 0; c < nn.weight[i].getColLength(); ++c) {
					nn.weight[i](r, c) = j["weights"][i][r][c].get<float>();
				}
			}
		}
		// Load Biases
		for (size_t i = 0; i <= nn.hiddenLayerLen; ++i) {
			for (size_t r = 0; r < nn.bias[i].getRowLength(); ++r) {
				for (size_t c = 0; c < nn.bias[i].getColLength(); ++c) {
					nn.bias[i](r, c) = j["bias"][i][r][c].get<float>();
				}
			}
		}
		return (nn);
	}
};

#endif
