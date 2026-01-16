#include "DMatrix.hpp"

DMatrix::DMatrix() : rows(2), cols(2), matrix(this->rows * this->cols, 0.0f) {
}

DMatrix::DMatrix(size_t row, size_t col)
	: rows(row), cols(col), matrix(row * col, 0.0f) {
}

DMatrix::DMatrix(const std::vector<float> &vectorArray)
	: rows(vectorArray.size()), cols(1), matrix(vectorArray) {
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

DMatrix DMatrix::multiplyVectorized(const DMatrix &other) const {
	DMatrix result(this->rows, other.cols);
	for (size_t i = 0; i < this->rows; i += BLOCK_SIZE) {
		for (size_t k = 0; k < this->cols; k += BLOCK_SIZE) {
			for (size_t j = 0; j < other.cols; j += BLOCK_SIZE) {
				const size_t maxii = std::min(i + BLOCK_SIZE, this->rows);
				for (size_t ii = i; ii < maxii; ++ii) {
					const size_t maxkk = std::min(k + BLOCK_SIZE, this->cols);
					for (size_t kk = k; kk < maxkk; ++kk) {
						const float	 a_val = this->matrix[ii * cols + kk];
						const size_t maxjj =
							std::min(j + BLOCK_SIZE, other.cols);
// Vectorization hints for compiler
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
						for (size_t jj = j; jj < maxjj; ++jj) {
							result.matrix[ii * result.cols + jj] +=
								a_val * other.matrix[kk * other.cols + jj];
						}
					}
				}
			}
		}
	}
	return result;
}

DMatrix DMatrix::multiplyOptimized(const DMatrix &other) const {
	DMatrix result(this->rows, other.cols);
	// Use smaller block size for better cache utilization on innermost loops
	constexpr size_t MICRO_BLOCK = 8;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
	for (size_t i = 0; i < this->rows; i += BLOCK_SIZE) {
		for (size_t j = 0; j < other.cols; j += BLOCK_SIZE) {
			// Clear the result block
			const size_t maxii = std::min(i + BLOCK_SIZE, this->rows);
			const size_t maxjj = std::min(j + BLOCK_SIZE, other.cols);
			for (size_t k = 0; k < this->cols; k += BLOCK_SIZE) {
				const size_t maxkk = std::min(k + BLOCK_SIZE, this->cols);
				// Micro-blocking for even better cache behavior
				for (size_t ii = i; ii < maxii; ii += MICRO_BLOCK) {
					for (size_t kk = k; kk < maxkk; kk += MICRO_BLOCK) {
						const size_t micro_ii_end =
							std::min(ii + MICRO_BLOCK, maxii);
						const size_t micro_kk_end =
							std::min(kk + MICRO_BLOCK, maxkk);
						for (size_t iii = ii; iii < micro_ii_end; ++iii) {
							for (size_t kkk = kk; kkk < micro_kk_end; ++kkk) {
								const float a_val =
									this->matrix[iii * cols + kkk];
								// Prefetch next row of B matrix
								if (kkk + 1 < micro_kk_end) {
									__builtin_prefetch(
										&other.matrix[(kkk + 1) * other.cols +
													  j],
										0, 1);
								}
// Vectorizable innermost loop
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable) unroll(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#pragma GCC unroll 4
#endif
								for (size_t jjj = j; jjj < maxjj; ++jjj) {
									result.matrix[iii * result.cols + jjj] +=
										a_val *
										other.matrix[kkk * other.cols + jjj];
								}
							}
						}
					}
				}
			}
		}
	}
	return result;
}

DMatrix DMatrix::operator*(const DMatrix &other) const {
	if (this->cols != other.rows) {
		throw std::invalid_argument(
			"Matrix dimensions incompatible for multiplication");
	}
	// Automatically choose best implementation based on size
	const size_t total_ops = this->rows * this->cols * other.cols;
	if (total_ops < 1000000) {
		return multiplyVectorized(other);
	}
	return multiplyOptimized(other);
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
