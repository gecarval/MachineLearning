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
	float learnRate;
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
	void train(const Vector2 *inputArray,
			   const size_t len,
			   const float desired) const;
	size_t getWeightArraySize(void) const;
};

#endif
