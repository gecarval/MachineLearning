#include "Perceptron.hpp"
#include <cstddef>

Perceptron::Perceptron()
	: bias(BIAS), learnRate(LEARNRATE), weightArray(new Vector2),
	  weightArraySize(1)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < this->weightArraySize; i++)
	{
		this->weightArray[i].x =
			(float) GetRandomValue(decimal, decimal) / decimal;
		this->weightArray[i].y =
			(float) GetRandomValue(decimal, decimal) / decimal;
	}
}

Perceptron::Perceptron(const size_t weightArrayLen)
	: bias(BIAS), learnRate(LEARNRATE),
	  weightArray(new Vector2[weightArrayLen]), weightArraySize(weightArrayLen)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < weightArraySize; i++)
	{
		this->weightArray[i].x = GetRandomValue(decimal, decimal) / decimal;
		this->weightArray[i].y = GetRandomValue(decimal, decimal) / decimal;
	}
}

Perceptron::~Perceptron()
{
	delete[] this->weightArray;
}

Perceptron::Perceptron(const Perceptron &other)
{
	if (this != &other)
		*this = other;
}

Perceptron &Perceptron::operator=(const Perceptron &other)
{
	if (this != &other)
	{
		this->bias = other.bias;
		this->learnRate = other.learnRate;
		const size_t len = other.getWeightArraySize();
		delete[] this->weightArray;
		this->weightArray = new Vector2[len];
		for (size_t i = 0; i < len; i++)
			this->weightArray[i] = other.weightArray[i];
		this->weightArraySize = other.getWeightArraySize();
	}
	return (*this);
}

float Perceptron::feedFoward(const Vector2 *inputArray, const size_t len) const
{
	Vector2 vecsum = Vector2Zero();
	for (size_t i = 0; i < len; i++)
		vecsum += inputArray[i] * this->weightArray[i];
	const float sum = vecsum.x + vecsum.y + this->bias;
	return (this->activate(sum));
}

float Perceptron::activate(const float sum) const
{
	float value;
	if (sum >= 0)
		value = 1;
	else
		value = -1;
	return (value);
}

void Perceptron::train(const Vector2 *inputArray,
					   const size_t len,
					   const float desired) const
{
	const float guess = this->feedFoward(inputArray, len);
	const float error = desired - guess;
	for (size_t i = 0; i < this->weightArraySize; i++)
		this->weightArray[i] =
			this->weightArray[i] + (inputArray[i] * this->learnRate * error);
}

size_t Perceptron::getWeightArraySize(void) const
{
	return (this->weightArraySize);
}
