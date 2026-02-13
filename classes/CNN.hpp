#ifndef CNN_HPP
#define CNN_HPP

#include "NeuralNetwork.hpp"
#include <iostream>
#include <system_error>
#include <vector>

class ConvNeuralNetwork {
  private:
	// Your existing Neural Network class handles the "Fully Connected" part
	NeuralNetwork classifier;

	// Attributes for convolutional layers
	size_t inputWidth;
	size_t inputHeight;
	size_t numFilters;
	size_t kernelSize;
	float  convLearnRate; // Separate learning rate for convolutional layers

	// Storage for convolutional kernels (Weights)
	std::vector<DMatrix> kernels;
	std::vector<float>	 kernelBiases;

	// Overloaded convolution methods
	DMatrix performConvolution(const DMatrix &inputImage) const;
	DMatrix performConvolution(const DMatrix		&inputImage,
							   std::vector<DMatrix> &featureMaps,
							   std::vector<DMatrix> &preActivation) const;

	// Backpropagation through convolutional layers
	void backpropConvolution(const DMatrix				&inputImage,
							 const std::vector<DMatrix> &featureMaps,
							 const std::vector<DMatrix> &preActivation,
							 const DMatrix				&errorFromFC);

  public:
	ConvNeuralNetwork();
	ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters, size_t kSize,
					  size_t outputNodes);
	ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters, size_t kSize,
					  size_t hiddenLayerLen, size_t outputNodes);

	std::vector<float> feedForward(const DMatrix &inputImage) const;
	void			   train(const DMatrix &inputImage, const DMatrix &target);

	void				 setClassifier(const NeuralNetwork &classifier);
	NeuralNetwork		&getClassifier();
	const NeuralNetwork &getClassifier() const;

	void  setConvLearnRate(float lr);
	float getConvLearnRate(void) const;

	// Access to convolutional layer parameters (for inspection/debugging)
	const std::vector<DMatrix> &getKernels() const;
	const std::vector<float>   &getKernelBiases() const;
};

#endif
