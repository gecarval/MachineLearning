#include "DMatrix.hpp"

DMatrix::DMatrix() : rows(2), cols(2), matrix(NULL) {
	this->matrix = new float *[this->rows];
	for (size_t i = 0; i < this->rows; i++) {
		this->matrix[i] = new float[this->cols];
		for (size_t j = 0; j < this->cols; j++) this->matrix[i][j] = 0;
	}
}

DMatrix::DMatrix(const size_t rows, const size_t cols)
	: rows(rows), cols(cols), matrix(NULL) {
	this->matrix = new float *[this->rows];
	for (size_t i = 0; i < this->rows; i++) {
		this->matrix[i] = new float[this->cols];
		for (size_t j = 0; j < this->cols; j++) this->matrix[i][j] = 0;
	}
}

DMatrix::DMatrix(const std::vector<float> &vectorArray)
	: rows(vectorArray.size()), cols(1), matrix(NULL) {
	this->matrix = new float *[this->rows];
	for (size_t i = 0; i < this->rows; i++) {
		this->matrix[i] = new float[this->cols];
		this->matrix[i][0] = vectorArray[i];
	}
}

DMatrix::~DMatrix() {
	for (size_t i = 0; i < this->rows; i++) delete[] this->matrix[i];
	delete[] this->matrix;
}

DMatrix::DMatrix(const DMatrix &other)
	: rows(other.rows), cols(other.cols), matrix(NULL) {
	if (this != &other) {
		*this = other;
	}
}

DMatrix &DMatrix::operator=(const DMatrix &other) {
	if (this != &other) {
		if (this->matrix != NULL) {
			for (size_t i = 0; i < this->rows; i++) delete[] this->matrix[i];
			delete[] this->matrix;
		}
		this->cols = other.cols;
		this->rows = other.rows;
		this->matrix = new float *[this->rows];
		for (size_t i = 0; i < this->rows; i++) {
			this->matrix[i] = new float[this->cols];
			for (size_t j = 0; j < this->cols; j++)
				this->matrix[i][j] = other.matrix[i][j];
		}
	}
	return (*this);
}

float *DMatrix::operator[](const size_t index) {
	const size_t accessY = index < this->rows ? index : this->rows;
	return (this->matrix[accessY]);
}

const float *DMatrix::operator[](const size_t index) const {
	const size_t accessY = index < this->rows ? index : this->rows;
	return (this->matrix[accessY]);
}

DMatrix &DMatrix::operator+=(const DMatrix &other) {
	if (this->cols != other.cols || this->rows != other.rows) return (*this);
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] += other.matrix[i][j];
	return (*this);
}

DMatrix &DMatrix::operator-=(const DMatrix &other) {
	if (this->cols != other.cols || this->rows != other.rows) return (*this);
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] -= other.matrix[i][j];
	return (*this);
}

DMatrix &DMatrix::operator*=(const DMatrix &other) {
	if (this->cols != other.rows) return (*this);
	DMatrix m(this->rows, other.cols);
	for (size_t i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			float sum = 0;
			for (size_t k = 0; k < this->cols; k++)
				sum += this->matrix[i][k] * other.matrix[k][j];
			m.matrix[i][j] = sum;
		}
	}
	*this = m;
	return (*this);
}

DMatrix DMatrix::operator+(const DMatrix &other) const {
	if (this->cols != other.cols || this->rows != other.rows) return (*this);
	DMatrix m(*this);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++)
			m.matrix[i][j] += other.matrix[i][j];
	return (m);
}

DMatrix DMatrix::operator-(const DMatrix &other) const {
	if (this->cols != other.cols || this->rows != other.rows) return (*this);
	DMatrix m(*this);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++)
			m.matrix[i][j] -= other.matrix[i][j];
	return (m);
}

DMatrix DMatrix::operator*(const DMatrix &other) const {
	if (this->cols != other.rows) return (DMatrix(this->rows, other.cols));
	DMatrix m(this->rows, other.cols);
	for (size_t i = 0; i < m.rows; i++) {
		for (size_t j = 0; j < m.cols; j++) {
			float sum = 0;
			for (size_t k = 0; k < this->cols; k++)
				sum += this->matrix[i][k] * other.matrix[k][j];
			m.matrix[i][j] = sum;
		}
	}
	return (m);
}

DMatrix &DMatrix::operator+=(const float n) {
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++) this->matrix[i][j] += n;
	return (*this);
}

DMatrix &DMatrix::operator-=(const float n) {
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++) this->matrix[i][j] -= n;
	return (*this);
}

DMatrix &DMatrix::operator*=(const float n) {
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++) this->matrix[i][j] *= n;
	return (*this);
}

DMatrix DMatrix::operator+(const float n) const {
	DMatrix m(*this);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++) m.matrix[i][j] += n;
	return (m);
}

DMatrix DMatrix::operator-(const float n) const {
	DMatrix m(*this);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++) m.matrix[i][j] -= n;
	return (m);
}

DMatrix DMatrix::operator*(const float n) const {
	DMatrix m(*this);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++) m.matrix[i][j] *= n;
	return (m);
}

std::vector<float> DMatrix::toVector(void) const {
	std::vector<float> v;
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++) v.push_back(this->matrix[i][j]);
	return (v);
}

DMatrix DMatrix::transpose(void) {
	DMatrix m(this->cols, this->rows);
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++) m.matrix[i][j] = this->matrix[j][i];
	return (m);
}

float DMatrix::totalSum(void) const {
	float sum = 0;
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++) sum += this->matrix[i][j];
	return (sum);
}

void DMatrix::randomize(void) {
	if (this->cols == 0) return;
	float std_dev = std::sqrt(2.0 / static_cast<float>(this->cols));
	std::random_device				rd;
	std::mt19937					generator(rd());
	std::normal_distribution<float> distribution(0.0, std_dev);
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] = distribution(generator);
}

void DMatrix::randomize(const size_t fan_in) {
	if (fan_in == 0) return;
	float			   std_dev = std::sqrt(2.0 / static_cast<float>(fan_in));
	std::random_device rd;
	std::mt19937	   generator(rd());
	std::normal_distribution<float> distribution(0.0, std_dev);
	for (size_t i = 0; i < this->rows; ++i)
		for (size_t j = 0; j < this->cols; ++j)
			this->matrix[i][j] = distribution(generator);
}

void DMatrix::map(float (*func)(float)) {
	for (size_t i = 0; i < this->rows; i++)
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] = func(this->matrix[i][j]);
}

void DMatrix::setValue(const size_t row, const size_t col, const float val) {
	const size_t accessY = row < this->rows ? row : this->rows;
	const size_t accessX = col < this->cols ? col : this->cols;
	this->matrix[accessY][accessX] = val;
}

float DMatrix::getValue(const size_t row, const size_t col) const {
	const size_t accessY = row < this->rows ? row : this->rows;
	const size_t accessX = col < this->cols ? col : this->cols;
	return (this->matrix[accessY][accessX]);
}

size_t DMatrix::getRowLength(void) const {
	return (this->rows);
}

size_t DMatrix::getColLength(void) const {
	return (this->cols);
}

std::ostream &operator<<(std::ostream &out, const DMatrix &m) {
	out << std::endl;
	for (size_t i = 0; i < m.getRowLength(); i++) {
		out << "Y" << i << " [";
		for (size_t j = 0; j < m.getColLength(); j++) {
			out << m[i][j];
			if (j + 1 < m.getColLength()) out << ", ";
		}
		out << "]" << std::endl;
	}
	return (out);
}
