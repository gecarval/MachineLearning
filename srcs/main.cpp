#include "../includes/Machine.hpp"

float calcDeclive(float m, float x, float d) {
	return (m * x + d);
}

void initPoints(Machine &machine) {
	machine.points = std::vector<Vector2>(initialPointAmount);
	machine.desired = std::vector<float>(initialPointAmount);
	for (size_t i = 0; i < initialPointAmount; i++) {
		const float randXPos =
			GetRandomValue(-GetScreenWidth(), GetScreenWidth());
		const float randYPos =
			GetRandomValue(-GetScreenHeight(), GetScreenHeight());
		machine.points[i] = (Vector2){randXPos, randYPos};
		const float lineY = calcDeclive(initialLine.m, randXPos, initialLine.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

void initEngine(Machine &machine) {
	InitWindow(windowWidth, windowHeight, windowTitle);
	const int monitorID = GetCurrentMonitor();
	const int dynamicWindowWidth = GetMonitorWidth(monitorID) * 0.9f;
	const int dynamicWindowHeight = GetMonitorHeight(monitorID) * 0.9f;
	SetWindowSize(dynamicWindowWidth, dynamicWindowHeight);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	machine.NN =
		NeuralNetwork(inputNodes, hiddenNodes, outputNodes, hiddenLayerLength);
	machine.NN.setLearnRate(learningRate);
	machine.camera = (Camera2D){offset, target, rotation, zoom};
	machine.state = STATE::MAINMENU;
	machine.line = initialLine;
	initPoints(machine);
}

void updateState(Machine &machine) {
	SetExitKey(KEY_ESCAPE);
	switch (machine.state) {
		case STATE::MAINMENU:
			machine.state = handleMainMenuState(machine);
			break;
		case STATE::PERCEPTRON:
			machine.state = handlePerceptronState(machine);
			break;
		case STATE::NEURALNETWORK:
			machine.state = handleNeuralNetworkState(machine);
			break;
		default:
			break;
	}
}

void endEngine(Machine &machine) {
	machine.points.clear();
	rlImGuiShutdown();
	CloseWindow();
}

int main(void) {
	std::time_t now = std::time(0);
	std::tm	   *local_time = std::localtime(&now);
	std::srand(local_time->tm_sec);
	SetRandomSeed(local_time->tm_sec);
	Machine machine;
	initEngine(machine);
	while (!WindowShouldClose()) updateState(machine);
	endEngine(machine);
	return (0);
}
