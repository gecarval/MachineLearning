#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "./DMatrix.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef NNLEARNRATE
#define NNLEARNRATE 0.0001f
#endif

class NeuralNetwork {
  private:
	float	 learnRate;
	size_t	 numberOfInputsNodes;
	size_t	 numberOfHiddenNodes;
	size_t	 numberOfOutputNodes;
	size_t	 hiddenLayerLen;
	DMatrix *weight;
	DMatrix *bias;

  public:
	static const int CLAMP = 10000;
	static const int TOLERANCE = 1000;
	static const int ALPHA = 1000;
	virtual ~NeuralNetwork();
	explicit NeuralNetwork();
	explicit NeuralNetwork(const size_t numberOfInputsNodes,
						   const size_t numberOfHiddenNodes,
						   const size_t numberOfOutputNodes);
	explicit NeuralNetwork(const size_t numberOfInputsNodes,
						   const size_t numberOfHiddenNodes,
						   const size_t numberOfOutputNodes,
						   const size_t hiddenLayerLength);
	explicit NeuralNetwork(const NeuralNetwork &other);
	NeuralNetwork	  &operator=(const NeuralNetwork &other);
	DMatrix			   feedFoward(const DMatrix &input) const;
	std::vector<float> feedFoward(const std::vector<float> &input) const;
	void			   train(const DMatrix &inputArray, const DMatrix &desired);
	void			   setLearnRate(const float newLearnRate);
	float			   getLearnRate(void) const;
	size_t			   getNumberOfInputsNodes(void) const;
	size_t			   getNumberOfHiddenNodes(void) const;
	size_t			   getNumberOfOutputsNodes(void) const;
};

#endif
