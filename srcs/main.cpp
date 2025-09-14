#include "../includes/Machine.hpp"

// Window Settings
static const char		  windowTitle[] = "Machine Learning";
static const unsigned int windowWidth = 800;
static const unsigned int windowHeight = 600;
static const unsigned int frameLimit = 120;

// Neural Network Settings
static const unsigned int inputNodes = 2;
static const unsigned int hiddenNodes = 2;
static const unsigned int outputNodes = 1;
static const unsigned int hiddenLayerLength = 1;
static const float		  learningRate = 0.0001f;

// Line Settings
static const Line initialLine = {500.0f, 0.5f, -200.0f};

// Points Settings
static const unsigned int initialPointAmount = 2000;

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
	// Init Raylib Window with no Logs
	SetTraceLogLevel(LOG_ERROR);
	InitWindow(windowWidth, windowHeight, windowTitle);

	// Dynamic Window Size
	const int monitorID = GetCurrentMonitor();
	const int dynamicWindowWidth = GetMonitorWidth(monitorID) * 0.9f;
	const int dynamicWindowHeight = GetMonitorHeight(monitorID) * 0.9f;
	SetWindowSize(dynamicWindowWidth, dynamicWindowHeight);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);

	// Machine Camera2D Settings
	const float		   posX = dynamicWindowWidth / 2.0f;
	const float		   posY = dynamicWindowHeight / 2.0f;
	const Vector2	   screenMiddle = (Vector2){posX, posY};
	const Vector2	   target = screenMiddle;
	const Vector2	   offset = screenMiddle;
	static const float rotation = 0.0f;
	static const float zoom = 1.0f;
	machine.camera = (Camera2D){offset, target, rotation, zoom};

	// Neural Network and Perceptron States initialization
	machine.NN =
		NeuralNetwork(inputNodes, hiddenNodes, outputNodes, hiddenLayerLength);
	machine.NN.setLearnRate(learningRate);
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
