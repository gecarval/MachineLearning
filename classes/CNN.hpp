#ifndef CNN_HPP
#define CNN_HPP

#include "NeuralNetwork.hpp"
#include <iterator>
#include <system_error>

class ConvNeuralNetwork {
  private:
	// Attributes for convolutional layers
	size_t inputWidth;
	size_t inputHeight;
	size_t numFilters;
	size_t filtersDepth;
	size_t kernelSize;
	float  convLearnRate;

	// Storage for convolutional kernels (Weights)
	std::vector<std::vector<DMatrix>> kernels;
	std::vector<std::vector<float>>	  kernelBiases;

	// Argmax positions from max pooling (for correct backprop per filter)
	// poolArgmax[f] stores flat indices into the pre-pool feature map
	mutable std::vector<std::vector<size_t>> poolArgmax;

	// Neural Network class handles the "Fully Connected" part
	NeuralNetwork classifier;

	// Overloaded convolution methods
	DMatrix performConvolution(const DMatrix &inputImage) const;
	DMatrix performConvolution(const DMatrix		&inputImage,
							   std::vector<DMatrix> &featureMaps,
							   std::vector<DMatrix> &preActivation) const;

	// Backpropagation through convolutional layers
	void backpropConvolution(const DMatrix				&inputImage,
							 const std::vector<DMatrix> &featureMaps,
							 const std::vector<DMatrix> &preActivation,
							 const DMatrix &errorFromFC, const size_t i);

	std::vector<float> feedForward(const DMatrix &inputImage) const;
	void			   train(const DMatrix &inputImage, const DMatrix &target);

  public:
	static constexpr float DefaultConvLearnRate = 0.001f;
	~ConvNeuralNetwork();
	ConvNeuralNetwork();
	ConvNeuralNetwork(const size_t imgW, const size_t imgH, const size_t kSize,
					  const size_t HiddenNodes, const size_t outputNodes);
	ConvNeuralNetwork(const size_t imgW, const size_t imgH,
					  const size_t filters, const size_t filtersDepth,
					  const size_t kSize, const size_t minKSize,
					  const size_t HiddenNodes, const size_t outputNodes,
					  const size_t hiddenLayerLen);
	ConvNeuralNetwork(const ConvNeuralNetwork &other);
	ConvNeuralNetwork &operator=(const ConvNeuralNetwork &other);

	std::vector<float> feedForward(const std::vector<float> &inputImage) const;
	void			   train(const std::vector<float> &inputImage,
							 const std::vector<float> &target);

	void				 setClassifier(const NeuralNetwork &classifier);
	NeuralNetwork		&getClassifier();
	const NeuralNetwork &getClassifier() const;

	DMatrix getConvOnAFilterFunnel(const DMatrix &inputImage,
								   const size_t	  filter,
								   const size_t	  depth) const;

	void  setConvLearnRate(float lr);
	float getConvLearnRate(void) const;

	size_t getInputWidth() const;
	size_t getInputHeight() const;
	size_t getNumFilters() const;
	size_t getFiltersDepth() const;
	size_t getKernelSize() const;

	// Access to convolutional layer parameters (for inspection/debugging)
	const std::vector<std::vector<DMatrix>> &getKernels() const;
	const std::vector<std::vector<float>>	&getKernelBiases() const;

	static nlohmann::json serialize(const ConvNeuralNetwork &cnn,
									const std::string &filename) noexcept {
		nlohmann::json jsonData;
		jsonData["inputWidth"] = cnn.getInputWidth();
		jsonData["inputHeight"] = cnn.getInputHeight();
		jsonData["numFilters"] = cnn.getNumFilters();
		jsonData["filtersDepth"] = cnn.getFiltersDepth();
		jsonData["kernelSize"] = cnn.getKernelSize();
		jsonData["convLearnRate"] = cnn.getConvLearnRate();
		for (size_t f = 0; f < cnn.getNumFilters(); f++) {
			for (size_t d = 0; d < cnn.getFiltersDepth(); d++) {
				const DMatrix				   &kernel = cnn.getKernels()[f][d];
				std::vector<std::vector<float>> kernelData(
					kernel.getRowLength(),
					std::vector<float>(kernel.getColLength()));
				for (size_t r = 0; r < kernel.getRowLength(); r++) {
					for (size_t c = 0; c < kernel.getColLength(); c++) {
						kernelData[r][c] = kernel(r, c);
					}
				}
				jsonData["kernels"][f][d] = kernelData;
				jsonData["kernelBiases"][f][d] = cnn.getKernelBiases()[f][d];
			}
		}
		jsonData["classifier"] =
			NeuralNetwork::serialize(cnn.getClassifier(), "");
		std::ofstream file(filename);
		if (file.is_open()) {
			file << jsonData.dump(4);
			file.close();
		} else {
			std::cerr << "Error: Could not open file for writing: " << filename
					  << std::endl;
		}
		return (jsonData);
	}

	static ConvNeuralNetwork deserialize(const std::string &filename) noexcept {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Error: Could not open file for reading: " << filename
					  << std::endl;
			return ConvNeuralNetwork();
		}
		nlohmann::json jsonData;
		file >> jsonData;
		file.close();

		size_t inputWidth = jsonData["inputWidth"];
		size_t inputHeight = jsonData["inputHeight"];
		size_t numFilters = jsonData["numFilters"];
		size_t filtersDepth = jsonData["filtersDepth"];
		size_t kernelSize = jsonData["kernelSize"];
		float  convLearnRate = jsonData["convLearnRate"];

		ConvNeuralNetwork cnn(inputWidth, inputHeight, numFilters, filtersDepth,
							  kernelSize, kernelSize, 0, 0, 0);
		cnn.setConvLearnRate(convLearnRate);
		for (size_t f = 0; f < numFilters; f++) {
			for (size_t d = 0; d < filtersDepth; d++) {
				const auto &kernelData = jsonData["kernels"][f][d];
				DMatrix		kernel(kernelSize, kernelSize);
				for (size_t r = 0; r < kernelSize; r++) {
					for (size_t c = 0; c < kernelSize; c++) {
						kernel(r, c) = kernelData[r][c];
					}
				}
				cnn.kernels[f][d] = kernel;
				cnn.kernelBiases[f][d] =
					jsonData["kernelBiases"][f][d].get<float>();
			}
		}
		cnn.setClassifier(
			ConvNeuralNetwork::classifierFromJson(jsonData["classifier"]));
		return (cnn);
	}

	static NeuralNetwork classifierFromJson(const nlohmann::json &j) {
		NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
						 j.at("numberOfHiddenNodes").get<size_t>(),
						 j.at("numberOfOutputNodes").get<size_t>(),
						 j.at("hiddenLayerLen").get<size_t>());
		nn.setLearnRate(j.at("learnRate").get<float>());
		// Load Weights
		for (size_t i = 0; i <= nn.getHiddenLayerLength(); ++i) {
			for (size_t r = 0; r < nn.getWeightAt(i).getRowLength(); ++r) {
				for (size_t c = 0; c < nn.getWeightAt(i).getColLength(); ++c) {
					nn.getWeightAt(i)(r, c) =
						j["weights"][i][r][c].get<float>();
				}
			}
		}
		// Load Biases
		for (size_t i = 0; i <= nn.getHiddenLayerLength(); ++i) {
			for (size_t r = 0; r < nn.getBiasAt(i).getRowLength(); ++r) {
				for (size_t c = 0; c < nn.getBiasAt(i).getColLength(); ++c) {
					nn.getBiasAt(i)(r, c) = j["bias"][i][r][c].get<float>();
				}
			}
		}
		return (nn);
	}
};

#endif
