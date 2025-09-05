#include "DMatrix.hpp"

DMatrix::DMatrix() : rows(2), cols(2), matrix(NULL) {
	this->matrix = new float*[this->rows];
	for (size_t i = 0; i < this->rows; i++)
	{
		this->matrix[i] = new float[this->cols];
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] = 0;
	}
}

DMatrix::DMatrix(const size_t rows, const size_t cols) : rows(rows), cols(cols), matrix(NULL) {
	this->matrix = new float*[this->rows];
	for (size_t i = 0; i < this->rows; i++)
	{
		this->matrix[i] = new float[this->cols];
		for (size_t j = 0; j < this->cols; j++)
			this->matrix[i][j] = 0;
	}
}

DMatrix::~DMatrix() {
	for (size_t i = 0; i < this->rows; i++)
		delete [] this->matrix[i];
	delete [] this->matrix;
}

DMatrix::DMatrix(const DMatrix &other) : rows(other.rows), cols(other.cols), matrix(NULL) {
    if (this != &other) {
        *this = other;
    }
}

DMatrix &DMatrix::operator=(const DMatrix &other) {
    if (this != &other) {
		if (this->matrix != NULL)
		{
			for (size_t i = 0; i < this->rows; i++)
				delete [] this->matrix[i];
			delete [] this->matrix;
		}
        this->cols = other.cols;
		this->rows = other.rows;
		this->matrix = new float*[this->rows];
		for (size_t i = 0; i < this->rows; i++)
		{
			this->matrix[i] = new float[this->cols];
			for (size_t j = 0; j < this->cols; j++)
				this->matrix[i][j] = other.matrix[i][j];
		}
    }
    return (*this);
}

float *DMatrix::operator[](const size_t index)
{
	const size_t accessY = index < this->rows ? index : index < 0 ? 0 : index;
	return (this->matrix[accessY]);
}

const float *DMatrix::operator[](const size_t index) const
{
	const size_t accessY = index < this->rows ? index : index < 0 ? 0 : index;
	return (this->matrix[accessY]);
}

void DMatrix::setValue(const size_t row, const size_t col, const float val)
{
	const size_t accessY = row < this->rows ? row : row < 0 ? 0 : row;
	const size_t accessX = col < this->cols ? col : col < 0 ? 0 : col;
	this->matrix[accessY][accessX] = val;
}

float DMatrix::getValue(const size_t row, const size_t col) const
{
	const size_t accessY = row < this->rows ? row : row < 0 ? 0 : row;
	const size_t accessX = col < this->cols ? col : col < 0 ? 0 : col;
	return (this->matrix[accessY][accessX]);
}

size_t DMatrix::getRowLength(void) const
{
	return (this->rows);
}

size_t DMatrix::getColLength(void) const
{
	return (this->cols);
}

std::ostream &operator<<(std::ostream &out, const DMatrix &m)
{
	out << std::endl;
	for (size_t i = 0; i < m.getRowLength(); i++)
	{
		out << "Y[" << i << "] -> ";
		for (size_t j = 0; j < m.getColLength(); j++)
		{
			out << m[i][j];
			if (j + 1 < m.getColLength())
				out << ", ";
		}
		out << std::endl;
	}
	return (out);
}
