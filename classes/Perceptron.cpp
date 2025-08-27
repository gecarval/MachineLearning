#include "Perceptron.hpp"
#include <cstddef>

Perceptron::Perceptron() :
	bias(1.0),
	weightArray(new Vector2),
	weightArraySize(1) {
    std::cout << "Perceptron Constructor" << std::endl;
	for (size_t i = 0; i < 1; i++) {
		this->weightArray[i].x = GetRandomValue(-10, 10);
		this->weightArray[i].y = GetRandomValue(-10, 10);
	}
}

Perceptron::Perceptron(const size_t weightArrayLen) :
	bias(1.0),
	weightArray(new Vector2[weightArrayLen]),
	weightArraySize(weightArrayLen) {
    std::cout << "Perceptron Constructor" << std::endl;
	for (size_t i = 0; i < weightArraySize; i++) {
		this->weightArray[i].x = GetRandomValue(-10, 10);
		this->weightArray[i].y = GetRandomValue(-10, 10);
	}
}

Perceptron::~Perceptron() {
    std::cout << "Perceptron Destructor" << std::endl;
	delete[] this->weightArray;
}

Perceptron::Perceptron(const Perceptron &other) {
    std::cout << "Perceptron Copy Constructor" << std::endl;
    if (this != &other) {
        *this = other;
    }
}

Perceptron &Perceptron::operator=(const Perceptron &other) {
    std::cout << "Perceptron Assigment Operator" << std::endl;
    if (this != &other) {
        delete[] this->weightArray;
		const size_t len = other.getWeightArraySize();
		this->weightArray = new Vector2[len];
		for (size_t i = 0; i < len; i++)
			this->weightArray[i] = other.weightArray[i];
		this->weightArraySize = other.getWeightArraySize();
		this->bias = other.bias;
    }
    return (*this);
}

float Perceptron::feedFoward(const Vector2 *inputArray, const size_t len) const
{
	Vector2 vecsum = Vector2Zero();
	for (size_t i = 0; i < len; i++)
		vecsum += inputArray[i] * this->weightArray[i];
	vecsum = vecsum * this->bias;
	const float sum = vecsum.x + vecsum.y;
	return (this->activate(sum));
}

float Perceptron::activate(const float sum) const {
	float value;
	if (sum > 0)
		value = 1;
	else
		value = -1;
	return (value);
}

size_t Perceptron::getWeightArraySize(void) const {
	return (this->weightArraySize);
}
