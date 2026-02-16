#ifndef CNN_HPP
#define CNN_HPP

#include "NeuralNetwork.hpp"
#include <cstddef>
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

  public:
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

	std::vector<float> feedForward(const DMatrix &inputImage) const;
	void			   train(const DMatrix &inputImage, const DMatrix &target);

	void				 setClassifier(const NeuralNetwork &classifier);
	NeuralNetwork		&getClassifier();
	const NeuralNetwork &getClassifier() const;

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

	/* Helper: Converte a NeuralNetwork interna para um objeto JSON
	static nlohmann::json classifierToJson(const NeuralNetwork &nn) {
		nlohmann::json j;
		j["learnRate"] = nn.getLearnRate();
		j["numberOfInputsNodes"] = nn.getNumberOfInputsNodes();
		j["numberOfHiddenNodes"] = nn.getNumberOfHiddenNodes();
		j["numberOfOutputNodes"] = nn.getNumberOfOutputsNodes();
		j["hiddenLayerLen"] = nn.getHiddenLayerLength();

		j["weights"] = nlohmann::json::array();
		j["bias"] = nlohmann::json::array();

		for (size_t i = 0; i <= nn.getHiddenLayerLength(); ++i) {
			j["weights"].push_back(nn.getWeightAt(i).toVector());
			j["bias"].push_back(nn.getBiasAt(i).toVector());
		}
		return j;
	}

	// Helper: Reconstrói a NeuralNetwork a partir de um objeto JSON
	static NeuralNetwork classifierFromJson(const nlohmann::json &j) {
		NeuralNetwork nn(j.at("numberOfInputsNodes").get<size_t>(),
						 j.at("numberOfHiddenNodes").get<size_t>(),
						 j.at("numberOfOutputNodes").get<size_t>(),
						 j.at("hiddenLayerLen").get<size_t>());
		nn.setLearnRate(j.at("learnRate").get<float>());

		for (size_t i = 0; i <= nn.getHiddenLayerLength(); ++i) {
			std::vector<float> w_data =
				j.at("weights").at(i).get<std::vector<float>>();
			std::vector<float> b_data =
				j.at("bias").at(i).get<std::vector<float>>();

			// Como o constructor DMatrix(vector) cria uma matriz coluna,
			// precisamos garantir que as dimensões batam com o que a NN espera
			DMatrix &w_ref = const_cast<DMatrix &>(nn.getWeightAt(i));
			DMatrix &b_ref = const_cast<DMatrix &>(nn.getBiasAt(i));

			// Reconstrói mantendo as dimensões originais de linhas/colunas
			size_t	rows = w_ref.getRowLength();
			size_t	cols = w_ref.getColLength();
			DMatrix restoredW(rows, cols);
			for (size_t r = 0; r < rows; ++r)
				for (size_t c = 0; c < cols; ++c)
					restoredW.setValue(r, c, w_data[r * cols + c]);

			w_ref = restoredW;
			b_ref = DMatrix(b_data); // Bias é geralmente vetor coluna, então
									 // DMatrix(vector) funciona
		}
		return nn;
	}

	static void serialize(const ConvNeuralNetwork &cnn,
						  const std::string		  &filename) {
		nlohmann::json j;

		// 1. Metadados da Camada Convolucional
		j["inputWidth"] = cnn.inputWidth;
		j["inputHeight"] = cnn.inputHeight;
		j["numFilters"] = cnn.numFilters;
		j["kernelSize"] = cnn.kernelSize;
		j["convLearnRate"] = cnn.convLearnRate;

		// 2. Kernels e Biases
		j["kernels"] = nlohmann::json::array();
		for (const auto &k : cnn.kernels) {
			// Armazenamos um objeto com os dados e as dimensões para evitar
			// erros na reconstrução
			nlohmann::json k_obj;
			k_obj["rows"] = k.getRowLength();
			k_obj["cols"] = k.getColLength();
			k_obj["data"] = k.toVector();
			j["kernels"].push_back(k_obj);
		}
		j["kernelBiases"] = cnn.kernelBiases;

		// 3. Serializar o Classificador NeuralNetwork
		j["classifier"] = cnn.classifierToJson(cnn.classifier);

		// Salvar arquivo
		std::ofstream out(filename);
		if (!out.is_open())
			throw std::runtime_error(
				"Não foi possível abrir o arquivo para salvar a CNN.");
		out << j.dump(4);
	}

	static ConvNeuralNetwork deserialize(const std::string &filename) {
		std::ifstream in(filename);
		if (!in.is_open())
			throw std::runtime_error(
				"Não foi possível abrir o arquivo para carregar a CNN.");

		nlohmann::json j;
		in >> j;

		// Criar instância vazia (usando o default constructor)
		ConvNeuralNetwork cnn;
		cnn.inputWidth = j.at("inputWidth").get<size_t>();
		cnn.inputHeight = j.at("inputHeight").get<size_t>();
		cnn.numFilters = j.at("numFilters").get<size_t>();
		cnn.kernelSize = j.at("kernelSize").get<size_t>();
		cnn.convLearnRate = j.at("convLearnRate").get<float>();
		cnn.kernelBiases = j.at("kernelBiases").get<std::vector<float>>();

		// Reconstruir Kernels
		cnn.kernels.clear();
		for (const auto &k_obj : j.at("kernels")) {
			size_t			   r = k_obj.at("rows").get<size_t>();
			size_t			   c = k_obj.at("cols").get<size_t>();
			std::vector<float> data =
				k_obj.at("data").get<std::vector<float>>();

			DMatrix k(r, c);
			for (size_t i = 0; i < r; ++i)
				for (size_t j_idx = 0; j_idx < c; ++j_idx)
					k.setValue(i, j_idx, data[i * c + j_idx]);

			cnn.kernels.push_back(k);
		}

		// Reconstruir o Classificador
		cnn.classifier = cnn.classifierFromJson(j.at("classifier"));

		return cnn;
	}*/
};

#endif
