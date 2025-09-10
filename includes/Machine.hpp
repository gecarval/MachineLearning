#ifndef MACHINE_HPP
#define MACHINE_HPP

// INCLUDES
#include "../classes/NeuralNetwork.hpp"
#include "../classes/Perceptron.hpp"
#include "./imgui/imgui.h"
#include "imgui/rlImGui.h"
#include <chrono>
#include <string>
#include <vector>

struct Line {
	float x;
	float m;
	float d;
};

enum STATE { MAINMENU, PERCEPTRON, NEURALNETWORK };

struct Machine {
	int					 state;
	Camera2D			 camera;
	std::vector<Vector2> points;
	std::vector<float>	 desired;
	Perceptron			 brain;
	NeuralNetwork		 NN;
	Line				 line;
};

void  renderImGui(Machine &machine);
void  engineInput(Machine &machine);
float calcDeclive(float m, float x, float d);

#endif
