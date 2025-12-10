#include "DMatrix.hpp"

DMatrix::DMatrix() : rows(2), cols(2), matrix(this->rows * this->cols, 0.0f) {
}

DMatrix::DMatrix(size_t row, size_t col)
	: rows(row), cols(col), matrix(row * col, 0.0f) {
}

DMatrix::DMatrix(const std::vector<float> &vectorArray)
	: rows(vectorArray.size()), cols(1) {
	this->matrix.reserve(this->rows * this->cols);
	this->matrix = vectorArray;
}

DMatrix::DMatrix(const DMatrix &other)
	: rows(other.rows), cols(other.cols), matrix(other.matrix) {
}

DMatrix::~DMatrix() {
}

DMatrix &DMatrix::operator=(const DMatrix &other) {
	if (this != &other) {
		this->rows = other.rows;
		this->cols = other.cols;
		this->matrix = other.matrix;
	}
	return (*this);
}

float &DMatrix::operator()(size_t row, size_t col) {
	if (row >= rows || col >= cols) {
		throw std::out_of_range("Matrix index out of bounds");
	}
	return (this->matrix[row * this->cols + col]);
}

const float &DMatrix::operator()(size_t row, size_t col) const {
	if (row >= this->rows || col >= this->cols) {
		throw std::out_of_range("Matrix index out of bounds");
	}
	return (this->matrix[row * this->cols + col]);
}

DMatrix DMatrix::operator+(const DMatrix &other) const {
	if (this->rows != other.rows || this->cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions mismatch operator+");
	}
	DMatrix result(this->rows, this->cols);
	std::transform(this->matrix.begin(), this->matrix.end(),
				   other.matrix.begin(), result.matrix.begin(),
				   std::plus<float>());
	return (result);
}

DMatrix DMatrix::operator-(const DMatrix &other) const {
	if (this->rows != other.rows || this->cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions mismatch operator-");
	}
	DMatrix result(this->rows, this->cols);
	std::transform(this->matrix.begin(), this->matrix.end(),
				   other.matrix.begin(), result.matrix.begin(),
				   std::minus<float>());
	return (result);
}

DMatrix DMatrix::operator*(const DMatrix &other) const {
	if (this->cols != other.rows) {
		throw std::invalid_argument(
			"Matrix dimensions incompatible for multiplication");
	}
	DMatrix result(this->rows, other.cols);
	for (size_t i = 0; i < this->rows; i += BLOCK_SIZE) {
		for (size_t j = 0; j < other.cols; j += BLOCK_SIZE) {
			for (size_t k = 0; k < this->cols; k += BLOCK_SIZE) {
				for (size_t ii = i; ii < std::min(i + BLOCK_SIZE, this->rows);
					 ++ii) {
					for (size_t jj = j;
						 jj < std::min(j + BLOCK_SIZE, other.cols); ++jj) {
						float sum = 0.0f;
						for (size_t kk = k;
							 kk < std::min(k + BLOCK_SIZE, this->cols); ++kk) {
							sum += this->matrix[ii * cols + kk] *
								   other.matrix[kk * other.cols + jj];
						}
						result.matrix[ii * result.cols + jj] += sum;
					}
				}
			}
		}
	}
	return (result);
}

DMatrix &DMatrix::operator+=(const DMatrix &other) {
	if (this->rows != other.rows || this->cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions mismatch operator+=");
	}
	std::transform(this->matrix.begin(), this->matrix.end(),
				   other.matrix.begin(), this->matrix.begin(),
				   std::plus<float>());
	return (*this);
}

DMatrix &DMatrix::operator-=(const DMatrix &other) {
	if (this->rows != other.rows || this->cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions mismatch operator-=");
	}
	std::transform(this->matrix.begin(), this->matrix.end(),
				   other.matrix.begin(), this->matrix.begin(),
				   std::minus<float>());
	return (*this);
}

DMatrix &DMatrix::operator*=(const DMatrix &other) {
	*this = *this * other;
	return (*this);
}

DMatrix DMatrix::operator+(float n) const {
	DMatrix result(this->rows, this->cols);
	std::transform(this->matrix.begin(), this->matrix.end(),
				   result.matrix.begin(), [n](float x) { return x + n; });
	return (result);
}

DMatrix DMatrix::operator-(float n) const {
	DMatrix result(this->rows, this->cols);
	std::transform(this->matrix.begin(), this->matrix.end(),
				   result.matrix.begin(), [n](float x) { return x - n; });
	return (result);
}

DMatrix DMatrix::operator*(float n) const {
	DMatrix result(this->rows, this->cols);
	std::transform(this->matrix.begin(), this->matrix.end(),
				   result.matrix.begin(), [n](float x) { return x * n; });
	return (result);
}

DMatrix &DMatrix::operator+=(float n) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(), [n](float x) { return x + n; });
	return (*this);
}

DMatrix &DMatrix::operator-=(float n) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(), [n](float x) { return x - n; });
	return (*this);
}

DMatrix &DMatrix::operator*=(float n) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(), [n](float x) { return x * n; });
	return (*this);
}

std::vector<float> DMatrix::toVector() const {
	return (this->matrix);
}

DMatrix DMatrix::transpose() const {
	DMatrix result(this->cols, this->rows);
	for (size_t i = 0; i < this->rows; ++i) {
		for (size_t j = 0; j < this->cols; ++j) {
			result.matrix[j * rows + i] = this->matrix[i * cols + j];
		}
	}
	return (result);
}

float DMatrix::totalSum() const {
	return (std::accumulate(this->matrix.begin(), this->matrix.end(), 0.0f));
}

void DMatrix::randomize() {
	static std::random_device rd;
	static std::mt19937		  generator(rd());
	float std_dev = std::sqrt(2.0f / static_cast<float>(cols));
	std::normal_distribution<float> distribution(0.0f, std_dev);
	std::generate(this->matrix.begin(), this->matrix.end(),
				  [&]() { return distribution(generator); });
}

void DMatrix::randomize(size_t fanIn) {
	if (fanIn == 0) return;
	static std::random_device rd;
	static std::mt19937		  generator(rd());
	float std_dev = std::sqrt(2.0f / static_cast<float>(fanIn));
	std::normal_distribution<float> distribution(0.0f, std_dev);
	std::generate(this->matrix.begin(), this->matrix.end(),
				  [&]() { return distribution(generator); });
}

void DMatrix::map(float (*func)(float)) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(), func);
}

void DMatrix::multiply(const DMatrix &other) {
	if (this->rows != other.rows || this->cols != other.cols) {
		throw std::invalid_argument("Matrix dimensions mismatch multiply()");
	}
	std::transform(this->matrix.begin(), this->matrix.end(),
				   other.matrix.begin(), this->matrix.begin(),
				   std::multiplies<float>());
}

void DMatrix::setValue(size_t row, size_t col, float val) {
	(*this)(row, col) = val;
}

float DMatrix::getValue(size_t row, size_t col) const {
	return ((*this)(row, col));
}

size_t DMatrix::getRowLength() const {
	return (this->rows);
}

size_t DMatrix::getColLength() const {
	return (this->cols);
}

std::ostream &operator<<(std::ostream &out, const DMatrix &m) {
	out << std::endl;
	for (size_t i = 0; i < m.getRowLength(); i++) {
		out << "Y" << i << " [";
		for (size_t j = 0; j < m.getColLength(); j++) {
			out << m(i, j);
			if (j + 1 < m.getColLength()) out << ", ";
		}
		out << "]" << std::endl;
	}
	return (out);
}
