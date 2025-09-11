#ifndef DMATRIX_HPP
#define DMATRIX_HPP

#include "JsonParser.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

class DMatrix {
  protected:
	size_t	rows;
	size_t	cols;
	float **matrix;

  public:
	virtual ~DMatrix();
	DMatrix();
	DMatrix(const std::vector<float> &vectorArray);
	DMatrix(const size_t rows, const size_t cols);
	DMatrix(const DMatrix &other);

	DMatrix &operator=(const DMatrix &other);
	DMatrix	 operator+(const DMatrix &other) const;
	DMatrix	 operator-(const DMatrix &other) const;
	DMatrix	 operator*(const DMatrix &other) const;
	DMatrix &operator+=(const DMatrix &other);
	DMatrix &operator-=(const DMatrix &other);
	DMatrix &operator*=(const DMatrix &other);

	DMatrix	 operator+(const float n) const;
	DMatrix	 operator-(const float n) const;
	DMatrix	 operator*(const float n) const;
	DMatrix &operator+=(const float n);
	DMatrix &operator-=(const float n);
	DMatrix &operator*=(const float n);

	float		*operator[](const size_t index);
	const float *operator[](const size_t index) const;

	std::vector<float> toVector(void) const;
	DMatrix			   transpose(void) const;
	float			   totalSum(void) const;
	void			   randomize(void);
	void			   randomize(const size_t fanIn);
	void			   map(float (*func)(float));
	void			   multiply(const DMatrix &other);
	void   setValue(const size_t row, const size_t col, const float val);
	float  getValue(const size_t row, const size_t col) const;
	size_t getRowLength(void) const;
	size_t getColLength(void) const;

	static void serialize(const DMatrix &matrix, const std::string &filename) {
		static const int indent = 4;
		std::ofstream	 out(filename);
		if (!out.is_open()) {
			throw std::runtime_error("Unable to open file for serialization: " +
									 filename);
		}
		JsonParser::writeObjectStart(out);
		JsonParser::writeKeyValue(out, "rows", matrix.getRowLength(), indent);
		JsonParser::writeKeyValue(out, "cols", matrix.getColLength(), indent);
		out << "  \"data\": [\n";
		for (size_t r = 0; r < matrix.getRowLength(); ++r) {
			JsonParser::writeArrayStart(out, indent * 2);
			for (size_t c = 0; c < matrix.getColLength(); ++c) {
				JsonParser::writeNumber(out, matrix.getValue(r, c),
										c < matrix.getColLength() - 1);
			}
			JsonParser::writeArrayEnd(out, indent * 2);
			if (r < matrix.getRowLength() - 1) out << ",";
			out << "\n";
		}
		out << "  ]\n";
		JsonParser::writeObjectEnd(out);
		out.close();
	}
	static DMatrix deserialize(const std::string &filename) {
		std::string content = JsonParser::readFile(filename);
		size_t		pos = 0;
		JsonParser::skipTo(content, pos, '{', "file root");
		pos = content.find("\"rows\":");
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'rows' not found");
		pos += std::string("\"rows\":").length();
		size_t rows = JsonParser::parseSizeT(content, pos, "rows");
		pos = content.find("\"cols\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'cols' not found");
		pos += std::string("\"cols\":").length();
		size_t	cols = JsonParser::parseSizeT(content, pos, "cols");
		DMatrix matrix(rows, cols);
		pos = content.find("\"data\":", pos);
		if (pos == std::string::npos)
			throw std::runtime_error("Key 'data' not found");
		pos += std::string("\"data\":").length();
		JsonParser::skipTo(content, pos, '[', "data array");
		for (size_t r = 0; r < rows; ++r) {
			JsonParser::skipTo(content, pos, '[', "row " + std::to_string(r));
			for (size_t c = 0; c < cols; ++c) {
				matrix.setValue(
					r, c,
					JsonParser::parseFloat(content, pos,
										   "data row " + std::to_string(r) +
											   " col " + std::to_string(c)));
				if (c < cols - 1) {
					JsonParser::skipTo(content, pos, ',',
									   "data row " + std::to_string(r));
				}
			}
			JsonParser::skipTo(content, pos, ']', "row " + std::to_string(r));
		}
		return (matrix);
	}
};

std::ostream &operator<<(std::ostream &out, const DMatrix &m);

#endif
