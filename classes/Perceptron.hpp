#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "../includes/raylib/raylib.h"
#include "../includes/raylib/raymath.h"
#include <cstddef>

#ifndef BIAS
#define BIAS 1.0f
#endif

#ifndef DECIMAL
#define DECIMAL 1000000.0f
#endif

#ifndef LEARNRATE
#define LEARNRATE 0.0000001f
#endif

class Perceptron
{
private:
	float bias;
	float biasWeight;
	float learnRate;
	Vector2 *weight;
	size_t weightLen;

protected:
	virtual float activate(const float sum) const;

public:
	virtual ~Perceptron();
	explicit Perceptron();
	explicit Perceptron(const size_t weightArrayLen);
	explicit Perceptron(const Perceptron &other);
	Perceptron &operator=(const Perceptron &other);
	float feedFoward(const Vector2 *inputArray, const size_t len) const;
	void train(const Vector2 &inputArray,
			   const size_t index,
			   const float desired);
	float getWeightedX0(const float x0, const size_t weightIndex) const;
	float getWeightedX1(const float x1, const size_t weightIndex) const;
	void setWeightAt(const Vector2 &newWeight, const size_t index);
	const Vector2 &getWeightAt(const size_t index) const;
	void setBias(const float newBiasWeight);
	float getBias(void) const;
	void setBiasWeight(const float newBias);
	float getBiasWeight(void) const;
	void setLearnRate(const float newLearnRate);
	float getLearnRate(void) const;
	size_t getWeightLen(void) const;
};

#endif
