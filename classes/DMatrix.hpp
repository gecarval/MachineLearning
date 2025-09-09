#ifndef DMATRIX_HPP
#define DMATRIX_HPP

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
	void   setValue(const size_t row, const size_t col, const float val);
	float  getValue(const size_t row, const size_t col) const;
	size_t getRowLength(void) const;
	size_t getColLength(void) const;
};

std::ostream &operator<<(std::ostream &out, const DMatrix &m);

#endif
