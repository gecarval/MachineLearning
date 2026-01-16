#ifndef DMATRIX_HPP
#define DMATRIX_HPP

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static const size_t BLOCK_SIZE = 64;

class DMatrix {
  protected:
	size_t			   rows;
	size_t			   cols;
	std::vector<float> matrix;

  public:
	DMatrix();
	DMatrix(size_t row, size_t col);
	DMatrix(const std::vector<float> &vectorArray);
	DMatrix(const DMatrix &other);
	~DMatrix();

	DMatrix &operator=(const DMatrix &other);
	DMatrix	 operator+(const DMatrix &other) const;
	DMatrix	 operator-(const DMatrix &other) const;
	DMatrix	 operator*(const DMatrix &other) const;
	DMatrix &operator+=(const DMatrix &other);
	DMatrix &operator-=(const DMatrix &other);
	DMatrix &operator*=(const DMatrix &other);

	DMatrix	 operator+(float n) const;
	DMatrix	 operator-(float n) const;
	DMatrix	 operator*(float n) const;
	DMatrix &operator+=(float n);
	DMatrix &operator-=(float n);
	DMatrix &operator*=(float n);

	float		&operator()(size_t row, size_t col);
	const float &operator()(size_t row, size_t col) const;

	std::vector<float> toVector() const;
	DMatrix			   transpose() const;
	float			   totalSum() const;
	void			   randomize();
	void			   randomize(size_t fanIn);
	void			   map(float (*func)(float));
	void			   multiply(const DMatrix &other);
	void			   pow(const float exp);
	void			   powneg(const float exp);
	void			   setValue(size_t row, size_t col, float val);
	float			   getValue(size_t row, size_t col) const;
	size_t			   getRowLength() const;
	size_t			   getColLength() const;
	// Different multiplication implementations
	DMatrix multiplyVectorized(const DMatrix &other) const;
	DMatrix multiplyOptimized(const DMatrix &other) const;
};

std::ostream &operator<<(std::ostream &out, const DMatrix &m);

#endif
