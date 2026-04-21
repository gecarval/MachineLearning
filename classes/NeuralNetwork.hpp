#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "../includes/json.hpp"
#include "./DMatrix.hpp"
#include <fstream>
#include <functional>

#ifndef NNLEARNRATE
#define NNLEARNRATE 0.001f
#endif

// ─────────────────────────── Gradient utilities ───────────────────────────

float clampGradient(float x);
float errorTolerance(float x);

// ─────────────────────────── Activation functions ─────────────────────────
// Every activation AND its derivative receives the PRE-activation value z.
// Exception: DSigmoid — receives z and computes σ(z)·(1−σ(z)) internally,
// so callers never need to branch on the function pointer identity.

float Tanh(float x);
float Sigmoid(float x);
float SiLU(float x);
float ReLU(float x);
float LeakyReLU(float x);
float Step(float x);
float Linear(float x);

// Derivatives (all receive the pre-activation z)
float DTanh(float x);
float DSigmoid(float x);
float DSiLU(float x);
float DReLU(float x);
float DLeakyReLU(float x);
float DStep(float x);
float DLinear(float x);

// Softmax (operates on a full column vector)
DMatrix Softmax(const DMatrix &x);
DMatrix DSoftmax(const DMatrix &output, const DMatrix &error);

// ──────────────────────────── NeuralNetwork ───────────────────────────────

class NeuralNetwork {
  protected:
	float  learnRate;
	size_t numberOfInputsNodes;
	size_t numberOfHiddenNodes;
	size_t numberOfOutputNodes;
	size_t hiddenLayerLen;

	std::vector<DMatrix> weight; // weight[i]: layer i weight matrix
	std::vector<DMatrix> bias;	 // bias[i]:   layer i bias column

	float (*HiddenActivate)(float);
	float (*HiddenDeactivate)(float);
	float (*OutputActivate)(float);
	float (*OutputDeactivate)(float);
	bool useSoftmax;

  public:
	// ── Constants ──────────────────────────────────────────────────────────
	static constexpr int CLAMP = 10000;
	static constexpr int TOLERANCE = 10000;
	static constexpr int ALPHA = 100;

	// ── Lifecycle ──────────────────────────────────────────────────────────
	virtual ~NeuralNetwork() = default;
	NeuralNetwork();
	NeuralNetwork(size_t inputs, size_t hidden, size_t outputs);
	NeuralNetwork(size_t inputs, size_t hidden, size_t outputs,
				  size_t hiddenLayers);
	NeuralNetwork(const NeuralNetwork &other) = default;
	NeuralNetwork &operator=(const NeuralNetwork &other) = default;

	// ── Inference ──────────────────────────────────────────────────────────
	DMatrix feedForward(const DMatrix &input) const;
	virtual std::vector<float>
	feedForward(const std::vector<float> &input) const;

	// ── Training ───────────────────────────────────────────────────────────
	virtual DMatrix train(const DMatrix &input, const DMatrix &desired);
	// Convenience overload used by the interactive state
	DMatrix train(const std::vector<float> &input,
				  const std::vector<float> &desired);

	void clampWeightsAndBiases();

	// ── Configuration ──────────────────────────────────────────────────────
	void setHiddenLayerActivation(float (*activate)(float),
								  float (*deactivate)(float));
	void setOutputLayerActivation(float (*activate)(float),
								  float (*deactivate)(float));
	void enableSoftmax(bool enable);
	void setLearnRate(float lr);

	// ── Accessors ──────────────────────────────────────────────────────────
	float  getLearnRate() const;
	size_t getHiddenLayerLength() const;
	size_t getNumberOfInputsNodes() const;
	size_t getNumberOfHiddenNodes() const;
	size_t getNumberOfOutputsNodes() const;

	DMatrix		  &getBiasAt(size_t index);
	DMatrix		  &getWeightAt(size_t index);
	const DMatrix &getBiasAt(size_t index) const;
	const DMatrix &getWeightAt(size_t index) const;

	// ── Evolutionary ───────────────────────────────────────────────────────
	virtual NeuralNetwork mutate(float (*func)(float)) const;

	// ── Serialization ──────────────────────────────────────────────────────
	static nlohmann::json serialize(const NeuralNetwork &nn,
									const std::string	&filename) noexcept;
	static NeuralNetwork  deserialize(const std::string &filename) noexcept;

  private:
	// Shared body used by all constructors — allocates and initialises
	// weight/bias matrices for a network with the current field values.
	void initLayers();
};

#endif // NEURALNETWORK_HPP
