#ifndef MACHINE_HPP
#define MACHINE_HPP

// INCLUDES
#include "../classes/NeuralNetwork.hpp"
#include "../classes/Perceptron.hpp"
#include "../classes/ui/Button.hpp"
#include "./imgui/imgui.h"
#include "imgui/rlImGui.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

struct Line {
	float x;
	float m;
	float d;
};

struct STATE {
	enum MENU { MAIN, SETTING };
	enum GAME { PERCEPTRON = 2, NEURALNETWORK, CNN };
};

struct Resolution {
	int width;
	int height;

	std::string toString() const {
		return std::to_string(width) + "x" + std::to_string(height);
	}
};

struct Machine {
	int					 state;
	Camera2D			 camera;
	std::vector<Vector2> points;
	std::vector<float>	 desired;
	Perceptron			 brain;
	NeuralNetwork		 NN;
	NeuralNetwork		 CNN;
	Line				 line;
};

// Universal Window Settings
static const Vector2 drawFpsPos = (Vector2){10.0f, 10.0f};

// Functions
float		  calcDeclive(float m, float x, float d);
Vector2		  createVector2(const float x, const float y);
Rectangle	  createRectangle(const Vector2 &pos, const Vector2 &size);
const Button &backButton(Machine &machine);
int			  handlePerceptronState(Machine &machine);
int			  handleNeuralNetworkState(Machine &machine);
int			  handleMenuState(Machine &machine);
int			  handleCNNState(Machine &machine);

#endif
