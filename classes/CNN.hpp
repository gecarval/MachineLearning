#ifndef CNN_HPP
#define CNN_HPP

#include "NeuralNetwork.hpp"
#include <fstream>
#include <stdexcept>

// =============================================================================
// ConvNeuralNetwork
//
// A standard multi-layer Convolutional Neural Network following the design of
// LeCun et al. (1998) and the implementation conventions of modern frameworks
// (PyTorch, TensorFlow):
//
//   • Forward operation per conv layer uses CROSS-CORRELATION (no kernel flip),
//     exactly as PyTorch nn.Conv2d and TF Conv2D do.
//
//   • Kernel layout:  kernels[layer][filter][inChannel]
//       layer     — conv layer index          (0 … numConvLayers-1)
//       filter    — output channel index      (0 … numFilters-1)
//       inChannel — input channel index       (0 … inChannels(layer)-1)
//         inChannels(layer == 0) = 1          (single-channel image input)
//         inChannels(layer  > 0) = numFilters (previous layer's output count)
//
//   • Bias layout: kernelBiases[layer][filter]
//       One scalar bias per output filter per layer (standard).
//
//   • Weight initialisation: He (Kaiming) uniform — N(0, sqrt(2/fanIn))
//       fanIn = kernelSize² × inChannels(layer)
//
//   • Activation: LeakyReLU after every conv layer (slope = 1/100 for x < 0).
//
//   • Pooling: single 2×2 max-pool applied ONCE after the last conv layer.
//     Argmax positions are recorded during the training forward pass and used
//     for gradient routing in backprop.
//
//   • Flattening: all filter maps (post-pool) are concatenated into a 1-D
//     vector and fed into a fully-connected NeuralNetwork classifier.
//
// ---------------------------------------------------------------------------
// Forward pass  (per layer l, per output filter f):
//
//   z[l][f]   = Σ_c  crossCorrelate( input[l][c],  K[l][f][c] )  +  b[l][f]
//   a[l][f]   = LeakyReLU( z[l][f] )
//
// where crossCorrelate is DMatrix::kernelMult() — slides kernel over input
// without flipping (standard industry convention).
//
// ---------------------------------------------------------------------------
// Backprop  (per layer l, deepest → shallowest):
//
//   delta[f]           =  errorMap[f]  ⊙  DLeakyReLU( z[l][f] )
//
//   dL/dK[l][f][c]     =  crossCorrelate( input[l][c],  delta[f] )
//                       =  input[l][c].kernelMult( delta[f] )          [no
//                       flip]
//
//   dL/db[l][f]        =  sum( delta[f] )
//
//   dL/dinput[l][c]    =  Σ_f  fullConvolution( delta[f],  K[l][f][c] )
//                       =  Σ_f  delta[f].convolveFullPadded( K[l][f][c] )
//                         (convolveFullPadded flips the kernel — correct
//                         gradient
//                          for a cross-correlation forward pass, see Goodfellow
//                          et al., Deep Learning, Ch. 9)
// =============================================================================

class ConvNeuralNetwork {
  private:
	// ───────────Hyper-parameters───────────
	size_t inputWidth;
	size_t inputHeight;
	size_t numFilters;	  // output channels per conv layer
	size_t numConvLayers; // number of stacked conv layers  (filtersDepth)
	size_t kernelSize; // spatial size of every kernel (square: kSize × kSize)
	float  convLearnRate;

	// ───────────Learnable parameters───────────

	// kernels[layer][filter][inChannel]  — 3-level hierarchy
	std::vector<std::vector<std::vector<DMatrix>>> kernels;
	// kernelBiases[layer][filter]        — one scalar per output filter per
	// layer
	std::vector<std::vector<float>> kernelBiases;

	// Training state (written only by the training forward pass)
	// poolArgmax[filter]: flat argmax indices from the final 2×2 max-pool step.
	// Never touched by the const inference path — NOT declared mutable.
	std::vector<std::vector<size_t>> poolArgmax;

	// Fully-connected classifier
	NeuralNetwork classifier;

	// ───────────Internal helpers───────────

	// Returns the number of input channels for a given conv layer.
	//   layer 0  →  1          (the original single-channel image)
	//   layer l  →  numFilters (output channel count of the previous layer)
	size_t inChannelsAt(size_t layer) const noexcept;

	// Computes the spatial size (width = height, both shrink equally) of the
	// feature maps that exit layer `layer`, before max-pooling.
	// Each conv layer (valid / no-padding) reduces the side by (kernelSize -
	// 1).
	size_t spatialSizeAfterConv(size_t layer) const noexcept;

	// Allocates and He-initialises kernels and biases for all conv layers.
	void initKernels();

	// Computes the total number of values in the flattened post-pool vector
	// that is fed to the FC classifier.
	size_t computeFlattenedSize() const;

	// ───────────Forward pass helpers───────────

	// Inference-only: conv layers → max-pool → flatten.
	// Returns a column DMatrix of shape (flattenedSize × 1).
	DMatrix forwardConv(const DMatrix &image) const;

	// Training: same pipeline but records all intermediates needed for
	// backprop.
	//
	//   layerInputs[l]    — input channel maps fed into conv layer l
	//                       (vector of DMatrix, one per input channel)
	//   preAct[l][f]      — pre-activation map z at (layer l, filter f)
	//
	// poolArgmax[f] is populated here (non-const via direct member write since
	// this function is called exclusively from the non-const train() path).
	//
	// Returns a column DMatrix of shape (flattenedSize × 1).
	DMatrix forwardConvTrain(const DMatrix					   &image,
							 std::vector<std::vector<DMatrix>> &layerInputs,
							 std::vector<std::vector<DMatrix>> &preAct);

	// ───────────Backward pass helper───────────

	// Backpropagates `fcGrad` (gradient from classifier.train(), shape
	// flattenedSize×1) through the max-pool and all conv layers, updating
	// kernels and biases in-place.
	void backwardConv(const DMatrix						&fcGrad,
					  std::vector<std::vector<DMatrix>> &layerInputs,
					  std::vector<std::vector<DMatrix>> &preAct);

	// ───────────Private DMatrix-level wrappers───────────

	// Convert a flat float vector to a
	// row-major image DMatrix.
	DMatrix vectorToImage(const std::vector<float> &v) const;

  public:
	// Default learning rate for the convolutional layers.
	static constexpr float DefaultConvLearnRate = 0.001f;

	// ───────────Lifecycle───────────

	ConvNeuralNetwork();
	~ConvNeuralNetwork();
	ConvNeuralNetwork(const ConvNeuralNetwork &other);
	ConvNeuralNetwork &operator=(const ConvNeuralNetwork &other);

	// Simple constructor — 1 conv layer, 3 filters, kSize×kSize kernels.
	ConvNeuralNetwork(size_t imgW, size_t imgH, size_t kSize,
					  size_t hiddenNodes, size_t outputNodes);

	// Full constructor — configurable depth, filter count, and FC topology.
	ConvNeuralNetwork(size_t imgW, size_t imgH, size_t filters,
					  size_t numConvLayers, size_t kSize, size_t hiddenNodes,
					  size_t outputNodes, size_t hiddenLayerLen);

	// ───────────Inference & training───────────

	std::vector<float> feedForward(const std::vector<float> &image) const;
	void			   train(const std::vector<float> &image,
							 const std::vector<float> &target);

	// ───────────Inspection───────────

	// Returns the feature map produced by applying conv layers 0…layer for the
	// given filter index. Useful for visualisation / debugging.
	DMatrix getConvOnAFilterFunnel(const DMatrix &image, size_t filter,
								   size_t layer) const;

	// ───────────Accessors───────────

	void				 setClassifier(const NeuralNetwork &c);
	NeuralNetwork		&getClassifier();
	const NeuralNetwork &getClassifier() const;

	void  setConvLearnRate(float lr);
	float getConvLearnRate() const;

	size_t getInputWidth() const;
	size_t getInputHeight() const;
	size_t getNumFilters() const;
	size_t getNumConvLayers() const;
	size_t getKernelSize() const;

	const std::vector<std::vector<std::vector<DMatrix>>> &getKernels() const;
	const std::vector<std::vector<float>> &getKernelBiases() const;

	// ───────────Serialisation───────────

	static nlohmann::json	 serialize(const ConvNeuralNetwork &cnn,
									   const std::string	   &filename) noexcept;
	static ConvNeuralNetwork deserialize(const std::string &filename) noexcept;
	static NeuralNetwork	 classifierFromJson(const nlohmann::json &j);
};

#endif // CNN_HPP
