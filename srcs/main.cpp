#include "../includes/Machine.hpp"

// Line Settings
static const Line initialLine = {500.0f, 0.5f, -200.0f};

float calcDeclive(float m, float x, float d) {
	return (m * x + d);
}

const Button &backButton(Machine &machine) {
	const Vector2 buttonSize = createVector2(100, 50);
	const Vector2 buttonPos = createVector2(GetScreenWidth() - buttonSize.x, 0);
	const Rectangle bounds = createRectangle(buttonPos, buttonSize);
	static Button	button(bounds, "Back");
	button.setPosition(buttonPos.x, buttonPos.y);
	button.setOnClick([&machine]() { machine.state = STATE::MENU::MAIN; });
	button.update();
	return (button);
}

Vector2 createVector2(const float x, const float y) {
	return ((Vector2){x, y});
}

Rectangle createRectangle(const Vector2 &pos, const Vector2 &size) {
	return ((Rectangle){pos.x, pos.y, size.x, size.y});
}

void initPoints(Machine &machine) {
	static const unsigned int initialPointAmount = 2000;
	machine.points.resize(initialPointAmount);
	machine.desired.resize(initialPointAmount);
	for (size_t i = 0; i < initialPointAmount; i++) {
		const float randXPos =
			GetRandomValue(-GetScreenWidth(), GetScreenWidth());
		const float randYPos =
			GetRandomValue(-GetScreenHeight(), GetScreenHeight());
		machine.points[i] = createVector2(randXPos, randYPos);
		const float lineY = calcDeclive(initialLine.m, randXPos, initialLine.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1.0f : -1.0f;
	}
}

void initEngine(Machine &machine) {
	// Window Settings
	constexpr char		   WINDOW_TITLE[] = "Machine Learning";
	constexpr unsigned int WINDOW_WIDTH = 800;
	constexpr unsigned int WINDOW_HEIGHT = 600;
	constexpr unsigned int FRAME_LIMIT = 120;
	// Init Raylib Window
	SetTraceLogLevel(
		LOG_WARNING); // Changed from LOG_ERROR to see important warnings
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE);
	SetTargetFPS(FRAME_LIMIT);
	rlImGuiSetup(true);
	// Camera2D Settings
	const Vector2 screenMiddle =
		createVector2(WINDOW_WIDTH / 2.0f, WINDOW_HEIGHT / 2.0f);
	machine.camera = (Camera2D){.offset = screenMiddle,
								.target = screenMiddle,
								.rotation = 0.0f,
								.zoom = 1.0f};
	// Neural Network Settings (for binary classification)
	constexpr unsigned int NN_INPUT_NODES = 2;
	constexpr unsigned int NN_HIDDEN_NODES = 6;
	constexpr unsigned int NN_OUTPUT_NODES = 1;
	constexpr unsigned int NN_HIDDEN_LAYERS = 3;
	constexpr float		   NN_LEARNING_RATE = 0.0001f;
	machine.NN = NeuralNetwork(NN_INPUT_NODES, NN_HIDDEN_NODES, NN_OUTPUT_NODES,
							   NN_HIDDEN_LAYERS);
	machine.NN.setLearnRate(NN_LEARNING_RATE);
	machine.NN.setOutputLayerActivation(Sigmoid, DSigmoid);
	// CNN Settings (for digit recognition - 10 classes)
	constexpr unsigned int CANVAS_WIDTH = 28;
	constexpr unsigned int CANVAS_HEIGHT = 28;
	constexpr unsigned int CNN_FILTERS = 8;
	constexpr unsigned int CNN_FILTERS_DEPHT = 1;
	constexpr unsigned int CNN_KERNEL_SIZE = 5;
	constexpr unsigned int CNN_MIN_KERNEL_SIZE = 5;
	constexpr unsigned int CNN_HIDDEN_NODES = 128;
	constexpr unsigned int CNN_OUTPUT_NODES = 10; // Digits 0-9
	constexpr unsigned int CNN_HIDDEN_LAYERS_LEN = 2;
	constexpr float		   CNN_LEARNING_RATE = 0.01f;
	constexpr float		   CNN_CONVLEARNING_RATE = 0.01f;
	machine.CNN = ConvNeuralNetwork(CANVAS_WIDTH, CANVAS_HEIGHT, CNN_FILTERS,
									CNN_FILTERS_DEPHT, CNN_KERNEL_SIZE,
									CNN_MIN_KERNEL_SIZE, CNN_HIDDEN_NODES,
									CNN_OUTPUT_NODES, CNN_HIDDEN_LAYERS_LEN);
	machine.CNN.getClassifier().setLearnRate(CNN_LEARNING_RATE);
	machine.CNN.setConvLearnRate(CNN_CONVLEARNING_RATE);
	// Initialize state and line
	machine.state = STATE::MENU::MAIN;
	machine.line = initialLine;
	// Initialize points for perceptron
	initPoints(machine);
	TraceLog(LOG_INFO, "Engine initialized successfully");
}

void updateState(Machine &machine) {
	SetExitKey(KEY_ESCAPE);
	switch (machine.state) {
		case STATE::MENU::MAIN:
		case STATE::MENU::SETTING:
			machine.state = handleMenuState(machine);
			break;
		case STATE::GAME::PERCEPTRON:
			machine.state = handlePerceptronState(machine);
			break;
		case STATE::GAME::NEURALNETWORK:
			machine.state = handleNeuralNetworkState(machine);
			break;
		case STATE::GAME::CNN:
			machine.state = handleCNNState(machine);
			break;
		default:
			TraceLog(LOG_WARNING, "Unknown state: %d, returning to main menu",
					 machine.state);
			machine.state = STATE::MENU::MAIN;
			break;
	}
}

void endEngine(Machine &machine) {
	machine.points.clear();
	machine.desired.clear();
	rlImGuiShutdown();
	CloseWindow();
	TraceLog(LOG_INFO, "Engine shutdown complete");
}

int main(void) {
	// Seed random number generators
	const std::time_t  now = std::time(nullptr);
	std::tm			  *localTime = std::localtime(&now);
	const unsigned int seed = static_cast<unsigned int>(
		localTime->tm_sec + localTime->tm_min * 60 + localTime->tm_hour * 3600);
	std::srand(seed);
	SetRandomSeed(seed);
	// Initialize and run
	Machine machine;
	initEngine(machine);
	while (!WindowShouldClose()) {
		updateState(machine);
	}
	endEngine(machine);
	return (0);
}
