#ifndef MACHINE_HPP
#define MACHINE_HPP

// INCLUDES
#include "../classes/Perceptron.hpp"
#include "./imgui/imgui.h"
#include "imgui/rlImGui.h"

struct Machine {
	Camera2D			 camera;
	std::vector<Vector2> points;
	Perceptron			 brain;
};

void renderImGui(Machine &machine);
void engineInput(Machine &machine);

#endif
