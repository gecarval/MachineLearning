#include "Perceptron.hpp"
#include <cstddef>

Perceptron::Perceptron()
	: bias(BIAS), learnRate(LEARNRATE), weight(new Vector2[1]), weightLen(1)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < this->weightLen; i++)
	{
		this->weight[i].x = GetRandomValue(decimal, decimal) / decimal;
		this->weight[i].y = GetRandomValue(decimal, decimal) / decimal;
	}
	this->biasWeight = GetRandomValue(decimal, decimal) / decimal;
}

Perceptron::Perceptron(const size_t weightLen)
	: bias(BIAS), learnRate(LEARNRATE), weight(new Vector2[weightLen]),
	  weightLen(weightLen)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < this->weightLen; i++)
	{
		this->weight[i].x = GetRandomValue(decimal, decimal) / decimal;
		this->weight[i].y = GetRandomValue(decimal, decimal) / decimal;
	}
	this->biasWeight = GetRandomValue(decimal, decimal) / decimal;
}

Perceptron::Perceptron(const size_t weightLen, const float newBias)
	: bias(newBias), learnRate(LEARNRATE), weight(new Vector2[weightLen]),
	  weightLen(1)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < this->weightLen; i++)
	{
		this->weight[i].x = GetRandomValue(decimal, decimal) / decimal;
		this->weight[i].y = GetRandomValue(decimal, decimal) / decimal;
	}
	this->biasWeight = GetRandomValue(decimal, decimal) / decimal;
}

Perceptron::Perceptron(const size_t weightLen,
					   const float newBias,
					   const float newLearnRate)
	: bias(newBias), learnRate(newLearnRate), weight(new Vector2[weightLen]),
	  weightLen(1)
{
	static const float decimal = DECIMAL;
	for (size_t i = 0; i < this->weightLen; i++)
	{
		this->weight[i].x = GetRandomValue(decimal, decimal) / decimal;
		this->weight[i].y = GetRandomValue(decimal, decimal) / decimal;
	}
	this->biasWeight = GetRandomValue(decimal, decimal) / decimal;
}

Perceptron::~Perceptron()
{
	delete[] this->weight;
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
		this->biasWeight = other.biasWeight;
		this->weightLen = other.weightLen;
		const size_t len = other.weightLen;
		delete[] this->weight;
		this->weight = new Vector2[len];
		for (size_t i = 0; i < len; i++)
			this->weight[i] = other.weight[i];
	}
	return (*this);
}

float Perceptron::feedFoward(const Vector2 *inputArray, const size_t len) const
{
	Vector2 vecsum = Vector2Zero();
	for (size_t i = 0; i < len; i++)
		vecsum += inputArray[i] * this->weight[i];
	const float wbias = this->bias * this->biasWeight;
	const float sum = vecsum.x + vecsum.y + wbias;
	return (this->activate(sum));
}

float Perceptron::activate(const float sum) const
{
	float value;
	if (sum > 0)
		value = 1;
	else
		value = -1;
	return (value);
}

void Perceptron::train(const Vector2 &input,
					   const size_t index,
					   const float desired)
{
	const size_t len = 1;
	const float guess = this->feedFoward(&input, len);
	const float error = desired - guess;
	this->setWeightAt(
		this->getWeightAt(index) + (input * this->learnRate * error), index);
	this->biasWeight = this->biasWeight + this->learnRate * this->bias * error;
}

float Perceptron::getWeightedX0(const float x0, const size_t weightIndex) const
{
	if (weightIndex >= this->weightLen)
		return (x0 * this->weight[this->weightLen - 1].x);
	return (x0 * this->weight[weightIndex].x);
}

float Perceptron::getWeightedX1(const float x1, const size_t weightIndex) const
{
	if (weightIndex >= this->weightLen)
		return (x1 * this->weight[this->weightLen - 1].y);
	return (x1 * this->weight[weightIndex].y);
}

void Perceptron::setWeightAt(const Vector2 &newWeight, const size_t index)
{
	if (index >= this->weightLen)
		this->weight[this->weightLen - 1] = newWeight;
	else
		this->weight[index] = newWeight;
}

const Vector2 &Perceptron::getWeightAt(const size_t index) const
{
	if (index >= this->weightLen)
		return (this->weight[this->weightLen - 1]);
	return (this->weight[index]);
}

void Perceptron::setBias(const float newBias)
{
	this->bias = newBias;
}

float Perceptron::getBias(void) const
{
	return (this->bias);
}

void Perceptron::setBiasWeight(const float newBiasWeight)
{
	this->biasWeight = newBiasWeight;
}

float Perceptron::getBiasWeight(void) const
{
	return (this->biasWeight);
}

void Perceptron::setLearnRate(const float newLearnRate)
{
	this->learnRate = newLearnRate;
}

float Perceptron::getLearnRate(void) const
{
	return (this->learnRate);
}

size_t Perceptron::getWeightLen(void) const
{
	return (this->weightLen);
}
