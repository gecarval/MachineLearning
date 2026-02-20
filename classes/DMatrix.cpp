#include "DMatrix.hpp"
#include <cstddef>
#include <stdexcept>

DMatrix::DMatrix() : rows(0), cols(0), matrix(0) {
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
	return (this->matrix[row * this->cols + col]);
}

const float &DMatrix::operator()(size_t row, size_t col) const {
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
	return (result);
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
	return (result);
}

DMatrix DMatrix::operator*(const DMatrix &other) const {
	if (this->cols != other.rows) {
		throw std::invalid_argument(
			"Matrix dimensions incompatible for multiplication");
	}
	// Automatically choose best implementation based on size
	const size_t total_ops = this->rows * this->cols * other.cols;
	if (total_ops < 4000000) {
		return (multiplyVectorized(other));
	}
	return (multiplyOptimized(other));
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

// Optimized maxPooling: raw pointer, no bounds checks, 2x2 special-cased
// with explicit 4-element compare to help compiler generate SIMD maxps.
DMatrix DMatrix::maxPooling(const unsigned int poolSize) const {
	if (poolSize == 0 || this->rows < poolSize || this->cols < poolSize) {
		throw std::runtime_error("Invalid pool size or matrix too small");
	}
	const size_t outRows = this->rows / poolSize;
	const size_t outCols = this->cols / poolSize;
	const size_t inCols = this->cols;
	DMatrix		 output(outRows, outCols);

	const float *__restrict__ inp = this->matrix.data();
	float *__restrict__ out = output.matrix.data();

	// Fast path for the common 2x2 case used everywhere in CNN
	if (poolSize == 2) {
		for (size_t i = 0; i < outRows; ++i) {
			const float *row0 = inp + (i * 2) * inCols;
			const float *row1 = inp + (i * 2 + 1) * inCols;
			float		*orow = out + i * outCols;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
			for (size_t j = 0; j < outCols; ++j) {
				const float a = row0[j * 2];
				const float b = row0[j * 2 + 1];
				const float c = row1[j * 2];
				const float d = row1[j * 2 + 1];
				const float ab = a > b ? a : b;
				const float cd = c > d ? c : d;
				orow[j] = ab > cd ? ab : cd;
			}
		}
		return (output);
	}

	// General path for other pool sizes
	for (size_t i = 0; i < outRows; ++i) {
		for (size_t j = 0; j < outCols; ++j) {
			float maxVal = inp[(i * poolSize) * inCols + (j * poolSize)];
			for (size_t pi = 0; pi < poolSize; ++pi) {
				const float *row =
					inp + (i * poolSize + pi) * inCols + j * poolSize;
				for (size_t pj = 0; pj < poolSize; ++pj) {
					if (row[pj] > maxVal) {
						maxVal = row[pj];
					}
				}
			}
			out[i * outCols + j] = maxVal;
		}
	}
	return (output);
}

// Optimized averagePooling: raw pointer, no bounds checks, 2x2 special-cased
// with explicit 4-sum to help compiler generate SIMD maxps.
DMatrix DMatrix::averagePooling(const unsigned int poolSize) const {
	if (poolSize == 0 || this->rows < poolSize || this->cols < poolSize) {
		throw std::runtime_error("Invalid pool size or matrix too small");
	}
	const size_t outRows = this->rows / poolSize;
	const size_t outCols = this->cols / poolSize;
	const size_t inCols = this->cols;
	DMatrix		 output(outRows, outCols);

	const float *__restrict__ inp = this->matrix.data();
	float *__restrict__ out = output.matrix.data();

	// Fast path for the common 2x2 case used everywhere in CNN
	if (poolSize == 2) {
		for (size_t i = 0; i < outRows; ++i) {
			const float *row0 = inp + (i * 2) * inCols;
			const float *row1 = inp + (i * 2 + 1) * inCols;
			float		*orow = out + i * outCols;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
			for (size_t j = 0; j < outCols; ++j) {
				const float a = row0[j * 2];
				const float b = row0[j * 2 + 1];
				const float c = row1[j * 2];
				const float d = row1[j * 2 + 1];
				orow[j] = (a + b + c + d) / 4.0f;
			}
		}
		return (output);
	}

	// General path for other pool sizes
	for (size_t i = 0; i < outRows; ++i) {
		for (size_t j = 0; j < outCols; ++j) {
			float sumVal = 0.0f;
			for (size_t pi = 0; pi < poolSize; ++pi) {
				const float *row =
					inp + (i * poolSize + pi) * inCols + j * poolSize;
				for (size_t pj = 0; pj < poolSize; ++pj) {
					sumVal += row[pj];
				}
			}
			out[i * outCols + j] =
				sumVal / static_cast<float>((poolSize * poolSize));
		}
	}
	return (output);
}

// Optimized kernelMult: direct pointer access, row-cached kernel rows,
// sequential output writes. Avoids bounds-checked operator() on hot path.
DMatrix DMatrix::kernelMult(const DMatrix &kernel) const {
	if (this->rows < kernel.rows || this->cols < kernel.cols) {
		throw std::runtime_error("Kernel Mult Mismatch");
	}
	const size_t outRows = this->rows - kernel.rows + 1;
	const size_t outCols = this->cols - kernel.cols + 1;
	const size_t kRows = kernel.rows;
	const size_t kCols = kernel.cols;
	const size_t inCols = this->cols;
	DMatrix		 output(outRows, outCols);

	const float *__restrict__ inp = this->matrix.data();
	const float *__restrict__ ker = kernel.matrix.data();
	float *__restrict__ out = output.matrix.data();

	for (size_t i = 0; i < outRows; ++i) {
		for (size_t j = 0; j < outCols; ++j) {
			float sum = 0.0f;
			// Walk kernel rows: each kernel row is a contiguous slice of
			// the input row starting at (i+ki, j). Both pointers are
			// sequential in their inner loop → SIMD-friendly.
			for (size_t ki = 0; ki < kRows; ++ki) {
				const float *__restrict__ inp_row = inp + (i + ki) * inCols + j;
				const float *__restrict__ ker_row = ker + ki * kCols;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
				for (size_t kj = 0; kj < kCols; ++kj) {
					sum += inp_row[kj] * ker_row[kj];
				}
			}
			out[i * outCols + j] = sum;
		}
	}
	return (output);
}

// Optimized kernelMultHalfPadded: same as above but pads inline without
// allocating a full padded matrix — avoids O(imgW*imgH) allocation and
// a second pass over the data.
DMatrix DMatrix::kernelMultHalfPadded(const DMatrix &kernel) const {
	if (this->rows < kernel.rows || this->cols < kernel.cols) {
		throw std::runtime_error("Kernel Mult Mismatch");
	}
	const size_t padR = kernel.rows / 2;
	const size_t padC = kernel.cols / 2;
	const size_t kRows = kernel.rows;
	const size_t kCols = kernel.cols;
	const size_t inRows = this->rows;
	const size_t inCols = this->cols;
	DMatrix		 output(inRows, inCols); // same-size output (same padding)

	const float *__restrict__ inp = this->matrix.data();
	const float *__restrict__ ker = kernel.matrix.data();
	float *__restrict__ out = output.matrix.data();

	for (size_t i = 0; i < inRows; ++i) {
		for (size_t j = 0; j < inCols; ++j) {
			float sum = 0.0f;
			for (size_t ki = 0; ki < kRows; ++ki) {
				// Map back to padded coordinate, check bounds inline
				const long si =
					static_cast<long>(i + ki) - static_cast<long>(padR);
				if (si < 0 || static_cast<size_t>(si) >= inRows) continue;
				const float *__restrict__ ker_row = ker + ki * kCols;
				const float *__restrict__ inp_row =
					inp + static_cast<size_t>(si) * inCols;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
				for (size_t kj = 0; kj < kCols; ++kj) {
					const long sj =
						static_cast<long>(j + kj) - static_cast<long>(padC);
					if (sj < 0 || static_cast<size_t>(sj) >= inCols) continue;
					sum += inp_row[static_cast<size_t>(sj)] * ker_row[kj];
				}
			}
			out[i * inCols + j] = sum;
		}
	}
	return (output);
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

void DMatrix::pow(const float exp) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(),
				   [exp](float x) { return std::pow(x, exp); });
}

void DMatrix::powneg(const float exp) {
	std::transform(this->matrix.begin(), this->matrix.end(),
				   this->matrix.begin(),
				   [exp](float x) { return std::pow(-x, exp); });
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
