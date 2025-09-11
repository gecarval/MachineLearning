#ifndef MACHINE_HPP
#define MACHINE_HPP

// INCLUDES
#include "../classes/NeuralNetwork.hpp"
#include "../classes/Perceptron.hpp"
#include "./imgui/imgui.h"
#include "imgui/rlImGui.h"
#include <chrono>

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

// Window Settings
static const char		  windowTitle[] = "Machine Learning";
static const unsigned int windowWidth = 800;
static const unsigned int windowHeight = 600;
static const unsigned int frameLimit = 120;
static const Vector2	  drawFpsPos = (Vector2){10.0f, 10.0f};

// Machine Camera2D Settings
static const float	 posX = windowWidth / 2.0f;
static const float	 posY = windowHeight / 2.0f;
static const Vector2 screenMiddle = (Vector2){posX, posY};
static const Vector2 target = screenMiddle;
static const Vector2 offset = screenMiddle;
static const float	 rotation = 0.0f;
static const float	 zoom = 1.0f;

// Neural Network Settings
static const unsigned int inputNodes = 2;
static const unsigned int hiddenNodes = 2;
static const unsigned int outputNodes = 1;
static const unsigned int hiddenLayerLength = 1;

// Render Texture Settings
static const Color backGroundColor = RAYWHITE;

// Simulation Settings
static const Line		  initialLine = {500.0f, 0.5f, -200.0f};
static const unsigned int initialPointAmount = 5000;
static const unsigned int pointRadius = 3;
static const Color		  pointColor = BLACK;

// Functions
float calcDeclive(float m, float x, float d);
int	  handlePerceptronState(Machine &machine);
int	  handleNeuralNetworkState(Machine &machine);
int	  handleMainMenuState(Machine &machine);

#endif
