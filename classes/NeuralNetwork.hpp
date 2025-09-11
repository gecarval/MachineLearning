#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "./DMatrix.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#ifndef NNLEARNRATE
#define NNLEARNRATE 0.001f
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
	static const int CLAMP = 100000;
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

	DMatrix			   feedFoward(const DMatrix &input) const;
	std::vector<float> feedFoward(const std::vector<float> &input) const;
	void			   train(const DMatrix &inputArray, const DMatrix &desired);

	void		   setLearnRate(const float newLearnRate);
	float		   getLearnRate(void) const;
	size_t		   getHiddenLayerLength(void) const;
	size_t		   getNumberOfInputsNodes(void) const;
	size_t		   getNumberOfHiddenNodes(void) const;
	size_t		   getNumberOfOutputsNodes(void) const;
	const DMatrix &getBiasAt(const size_t index) const;
	const DMatrix &getWeigthAt(const size_t index) const;

	// Serialize NeuralNetwork to a JSON-like text file
	static void serialize(const NeuralNetwork &nn,
						  const std::string	  &filename) {
		std::ofstream out(filename);
		if (!out.is_open()) {
			throw std::runtime_error("Unable to open file for serialization: " +
									 filename);
		}

		JsonParser::writeObjectStart(out, 0);
		JsonParser::writeKeyValue(out, "learnRate", nn.getLearnRate(), 2);
		JsonParser::writeKeyValue(out, "numberOfInputsNodes",
								  nn.getNumberOfInputsNodes(), 2);
		JsonParser::writeKeyValue(out, "numberOfHiddenNodes",
								  nn.getNumberOfHiddenNodes(), 2);
		JsonParser::writeKeyValue(out, "numberOfOutputNodes",
								  nn.getNumberOfOutputsNodes(), 2);
		JsonParser::writeKeyValue(out, "hiddenLayerLen", nn.hiddenLayerLen, 2);

		out << "  \"weights\": [\n";
		for (size_t i = 0; i < nn.hiddenLayerLen + 1; ++i) {
			JsonParser::writeArrayStart(out, 4);
			for (size_t r = 0; r < nn.weight[i].getRowLength(); ++r) {
				JsonParser::writeArrayStart(out, 6);
				for (size_t c = 0; c < nn.weight[i].getColLength(); ++c) {
					JsonParser::writeNumber(out, nn.weight[i].getValue(r, c),
											c < nn.weight[i].getColLength() -
													1);
				}
				JsonParser::writeArrayEnd(out, 6);
				if (r < nn.weight[i].getRowLength() - 1) out << ",";
				out << "\n";
			}
			JsonParser::writeArrayEnd(out, 4);
			if (i < nn.hiddenLayerLen) out << ",";
			out << "\n";
		}
		out << "  ],\n";

		out << "  \"biases\": [\n";
		for (size_t i = 0; i < nn.hiddenLayerLen + 1; ++i) {
			JsonParser::writeArrayStart(out, 4);
			for (size_t r = 0; r < nn.bias[i].getRowLength(); ++r) {
				JsonParser::writeArrayStart(out, 6);
				for (size_t c = 0; c < nn.bias[i].getColLength(); ++c) {
					JsonParser::writeNumber(out, nn.bias[i].getValue(r, c),
											c < nn.bias[i].getColLength() - 1);
				}
				JsonParser::writeArrayEnd(out, 6);
				if (r < nn.bias[i].getRowLength() - 1) out << ",";
				out << "\n";
			}
			JsonParser::writeArrayEnd(out, 4);
			if (i < nn.hiddenLayerLen) out << ",";
			out << "\n";
		}
		out << "  ]\n";
		JsonParser::writeObjectEnd(out, 0);

		out.close();
	}

	// Deserialize NeuralNetwork from a JSON-like text file
	static NeuralNetwork deserialize(const std::string &filename) {
		std::string content = JsonParser::readFile(filename);
		size_t		pos = 0;

		JsonParser::skipTo(content, pos, '{', "file root");

		pos = content.find("\"learnRate\":");
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'learnRate' not found");
		pos += std::string("\"learnRate\":").length();
		float learnRate = JsonParser::parseFloat(content, pos, "learnRate");

		pos = content.find("\"numberOfInputsNodes\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'numberOfInputsNodes' not found");
		pos += std::string("\"numberOfInputsNodes\":").length();
		size_t numberOfInputsNodes =
			JsonParser::parseSizeT(content, pos, "numberOfInputsNodes");

		pos = content.find("\"numberOfHiddenNodes\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'numberOfHiddenNodes' not found");
		pos += std::string("\"numberOfHiddenNodes\":").length();
		size_t numberOfHiddenNodes =
			JsonParser::parseSizeT(content, pos, "numberOfHiddenNodes");

		pos = content.find("\"numberOfOutputNodes\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'numberOfOutputNodes' not found");
		pos += std::string("\"numberOfOutputNodes\":").length();
		size_t numberOfOutputNodes =
			JsonParser::parseSizeT(content, pos, "numberOfOutputNodes");

		pos = content.find("\"hiddenLayerLen\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'hiddenLayerLen' not found");
		pos += std::string("\"hiddenLayerLen\":").length();
		size_t hiddenLayerLen =
			JsonParser::parseSizeT(content, pos, "hiddenLayerLen");

		NeuralNetwork nn(numberOfInputsNodes, numberOfHiddenNodes,
						 numberOfOutputNodes, hiddenLayerLen);
		nn.setLearnRate(learnRate);

		auto parseMatrixArray = [&](const std::string &key, DMatrix *matrices) {
			pos = content.find("\"" + key + "\":", pos);
			if (pos == std::string::npos)
				throw std::runtime_error("Key '" + key + "' not found");
			pos += std::string("\"" + key + "\":").length();
			JsonParser::skipTo(content, pos, '[', key + " array");

			for (size_t i = 0; i < hiddenLayerLen + 1; ++i) {
				JsonParser::skipTo(content, pos, '[',
								   key + " matrix " + std::to_string(i));
				for (size_t r = 0; r < matrices[i].getRowLength(); ++r) {
					JsonParser::skipTo(content, pos, '[',
									   key + " matrix " + std::to_string(i) +
										   " row " + std::to_string(r));
					for (size_t c = 0; c < matrices[i].getColLength(); ++c) {
						matrices[i].setValue(
							r, c,
							JsonParser::parseFloat(
								content, pos,
								key + " matrix " + std::to_string(i) + " row " +
									std::to_string(r) + " col " +
									std::to_string(c)));
						if (c < matrices[i].getColLength() - 1) {
							JsonParser::skipTo(content, pos, ',',
											   key + " matrix " +
												   std::to_string(i) + " row " +
												   std::to_string(r));
						}
					}
					JsonParser::skipTo(content, pos, ']',
									   key + " matrix " + std::to_string(i) +
										   " row " + std::to_string(r));
				}
				JsonParser::skipTo(content, pos, ']',
								   key + " matrix " + std::to_string(i));
			}
		};

		parseMatrixArray("weights", nn.weight);
		parseMatrixArray("biases", nn.bias);

		return (nn);
	}
};

#endif
