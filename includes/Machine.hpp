#ifndef MACHINE_HPP
#define MACHINE_HPP

// INCLUDES
#include "../classes/NeuralNetwork.hpp"
#include "../classes/Perceptron.hpp"
#include "../classes/ui/Button.hpp"
#include "./imgui/imgui.h"
#include "imgui/rlImGui.h"
#include <chrono>

struct Line {
	float x;
	float m;
	float d;
};

struct STATE {
	enum MENU { MAIN, SETTING };
	enum GAME { PERCEPTRON = 2, NEURALNETWORK, CNN };
};

struct Machine {
	int					 state;
	Camera2D			 camera;
	std::vector<Vector2> points;
	std::vector<float>	 desired;
	Perceptron			 brain;
	NeuralNetwork		 NN;
	Line				 line;
};

// Universal Window Settings
static const Vector2 drawFpsPos = (Vector2){10.0f, 10.0f};

// Functions
float calcDeclive(float m, float x, float d);
int	  handlePerceptronState(Machine &machine);
int	  handleNeuralNetworkState(Machine &machine);
int	  handleMenuState(Machine &machine);

#endif
