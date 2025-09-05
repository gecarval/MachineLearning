#ifndef DMATRIX_HPP
#define DMATRIX_HPP

#include <cstddef>
#include <ostream>

class DMatrix {
protected:
	size_t	rows;
	size_t	cols;
	float	**matrix;

public:
    virtual ~DMatrix();
    explicit DMatrix();
    explicit DMatrix(const size_t rows, const size_t cols);
    explicit DMatrix(const DMatrix &other);
    DMatrix &operator=(const DMatrix &other);
	float *operator[](const size_t index);
	const float *operator[](const size_t index) const;
	void setValue(const size_t row, const size_t col, const float val);
	float getValue(const size_t row, const size_t col) const;
	size_t getRowLength(void) const;
	size_t getColLength(void) const;
};

std::ostream &operator<<(std::ostream &out, const DMatrix &m);

#endif
