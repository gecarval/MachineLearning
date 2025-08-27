#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "../includes/raylib/raylib.h"
#include "../includes/raylib/raymath.h"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

class Perceptron {
private:
	float bias;
	Vector2 *weightArray;
	size_t weightArraySize;

protected:
	virtual float activate(const float sum) const;

public:
    virtual ~Perceptron();
    explicit Perceptron();
    explicit Perceptron(const size_t weightArrayLen);
    explicit Perceptron(const Perceptron &other);
    Perceptron &operator=(const Perceptron &other);
	float feedFoward(const Vector2 *inputArray, const size_t len) const;
	size_t getWeightArraySize(void) const;
};


#endif
