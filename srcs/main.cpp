#include "../includes/Machine.hpp"

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

float calcDeclive(float m, float x, float d) {
	return (m * x + d);
}

void initPoints(Machine &machine) {
	machine.points = std::vector<Vector2>(initialPointAmount);
	machine.desired = std::vector<float>(initialPointAmount);
	for (size_t i = 0; i < initialPointAmount; i++) {
		const float randXPos = GetRandomValue(-windowWidth, windowWidth);
		const float randYPos = GetRandomValue(-windowHeight, windowHeight);
		machine.points[i] = (Vector2){randXPos, randYPos};
		const float lineY = calcDeclive(initialLine.m, randXPos, initialLine.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

void initEngine(Machine &machine) {
	try {
		machine.NN = NeuralNetwork::deserialize("NeuralNetwork.json");
	} catch (const std::runtime_error &error) {
		std::cerr << "[WARNING]" << std::endl;
		std::cerr << error.what() << std::endl;
		machine.NN = NeuralNetwork(inputNodes, hiddenNodes, outputNodes,
								   hiddenLayerLength);
	}
	machine.camera = (Camera2D){offset, target, rotation, zoom};
	machine.state = STATE::MAINMENU;
	machine.line = initialLine;
	InitWindow(windowWidth, windowHeight, windowTitle);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	initPoints(machine);
}

static float timer = 0;

void trainMachineNeuralNetwork(Machine &machine) {
	static const int   samples = 4;
	static const int   trainLen = 2;
	static const int   resLen = 1;
	static const float trainingData[samples][trainLen] = {
		{0, 0},
		{1, 1},
		{1, 0},
		{0, 1},
	};
	static const float trainingResult[samples][resLen] = {
		{0},
		{0},
		{1},
		{1},
	};
	const int		   randomIndex = GetRandomValue(0, samples - 1);
	std::vector<float> train;
	std::vector<float> res;
	for (int i = 0; i < trainLen; i++)
		train.push_back(trainingData[randomIndex][i]);
	for (int i = 0; i < resLen; i++)
		res.push_back(trainingResult[randomIndex][i]);
	machine.NN.train(train, res);
	if (timer > 1.0) {
		for (int i = 0; i < samples; i++) {
			const std::vector<float> trained = {trainingData[i][0],
												trainingData[i][1]};
			const std::vector<float> rest = {trainingResult[i][0]};
			std::cout << "Model Train Result:" << trained << rest
					  << machine.NN.feedFoward(trained);
		}
		timer = 0;
		NeuralNetwork::serialize(machine.NN, "NeuralNetwork.json");
	}
}

void renderNeuralNetwork(Machine &machine) {
	static const int trainLoop = 1000;
	for (int i = 0; i < trainLoop; i++) trainMachineNeuralNetwork(machine);
	static const float grid = 10;
	static const float cols = windowWidth / grid;
	static const float rows = windowHeight / grid;
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			const float				 x0 = i / cols;
			const float				 x1 = j / rows;
			const std::vector<float> input = {x0, x1};
			const float				 x = i * grid;
			const float				 y = j * grid;
			const Vector2			 gridPos = (Vector2){x, y};
			const Vector2			 gridSize = (Vector2){grid, grid};
			const std::vector<float> output = machine.NN.feedFoward(input);
			const unsigned char		 alpha = Remap(output[0], 0, 1, 0, 255);
			const Color gridColor = (Color){alpha, alpha, alpha, 255};
			DrawRectangleV(gridPos, gridSize, gridColor);
		}
	}
	timer += GetFrameTime();
}

void DrawPerceptron(Machine &machine) {
	// Constants for layout
	static const float	 thickness = 1.5f;
	static const Vector2 mockTest = Vector2Zero();
	// Position in top-right corner
	static const Vector2 perceptronOffset =
		(Vector2){GetScreenWidth() - 250.0f, 50.0f};
	static const Vector2 perceptronCenter =
		(Vector2){10.0f, 10.0f} + perceptronOffset;
	static const float perceptronRadius = 15.0f;
	// Increased spacing for input/output lines
	static const Vector2 w0LineOffset = (Vector2){perceptronRadius * -4, 30};
	static const Vector2 w1LineOffset = (Vector2){perceptronRadius * -4, -30};
	static const Vector2 yLineOffset = (Vector2){perceptronRadius * 4, 0};
	static const Vector2 w0LinePos = perceptronCenter + w0LineOffset;
	static const Vector2 w1LinePos = perceptronCenter + w1LineOffset;
	static const Vector2 yLinePos = perceptronCenter + yLineOffset;
	// Get perceptron values
	const float b = machine.brain.getBias();
	const float bh = machine.brain.getBias() / 2;
	const float bw = machine.brain.getBiasWeight();
	const float bwb = b * bw;
	const float w0 = machine.brain.getWeightedX0(1, 0);
	const float w1 = machine.brain.getWeightedX1(1, 0);
	const float y = machine.brain.feedFoward(&mockTest, 1);
	// Color calculations
	const unsigned char redShift =
		(unsigned char)Remap(w0, -1.0f, 1.0f, -190, 190);
	const unsigned char blueShift =
		(unsigned char)Remap(w1, -1.0f, 1.0f, -190, 190);
	const unsigned char biasShift =
		(unsigned char)Remap(bwb, -bh, bh, -255, 255);
	const unsigned char biasReverseShift =
		(unsigned char)Remap(bwb, bh, -bh, 0, 255);
	const unsigned char yShift = (unsigned char)Remap(y, -1, 1, 0, 255);
	// Define colors
	const Color inputX0PerceptronColor = (Color){redShift, 70, 70, 255};
	const Color inputX1PerceptronColor = (Color){70, 70, blueShift, 255};
	const Color innerPerceptronColor =
		(Color){biasReverseShift, biasReverseShift, biasReverseShift, 255};
	const Color outerPerceptronColor =
		(Color){biasShift, biasShift, biasShift, 255};
	const Color outputPerceptronColor = (Color){50, yShift, 50, 255};
	// Input lines
	DrawLineEx(w0LinePos, perceptronCenter, thickness, inputX0PerceptronColor);
	DrawLineEx(w1LinePos, perceptronCenter, thickness, inputX1PerceptronColor);
	// Output line
	DrawLineEx(perceptronCenter, yLinePos, thickness, outputPerceptronColor);
	// Perceptron circles
	DrawCircleV(perceptronCenter, perceptronRadius + 5, outerPerceptronColor);
	DrawCircleV(perceptronCenter, perceptronRadius, innerPerceptronColor);
	// Input and output nodes
	DrawCircleV(w0LinePos, perceptronRadius * 0.5f, inputX0PerceptronColor);
	DrawCircleV(w1LinePos, perceptronRadius * 0.5f, inputX1PerceptronColor);
	DrawCircleV(yLinePos, perceptronRadius * 0.5f, outputPerceptronColor);
	// Labels
	DrawText("x0", w0LinePos.x - 30, w0LinePos.y - 10, 15,
			 inputX0PerceptronColor);
	DrawText("x1", w1LinePos.x - 30, w1LinePos.y - 10, 15,
			 inputX1PerceptronColor);
	DrawText("y", yLinePos.x + 15, yLinePos.y - 10, 15, outputPerceptronColor);
	DrawText("Perceptron", perceptronCenter.x - 40, perceptronCenter.y - 30, 15,
			 WHITE);
	// Weight and bias values
	DrawText(TextFormat("w0: %.2f", w0), w0LinePos.x + 10, w0LinePos.y + 10, 12,
			 inputX0PerceptronColor);
	DrawText(TextFormat("w1: %.2f", w1), w1LinePos.x + 10, w1LinePos.y - 20, 12,
			 inputX1PerceptronColor);
	DrawText(TextFormat("b: %.2f", bwb), perceptronCenter.x - 20,
			 perceptronCenter.y + 20, 12, outerPerceptronColor);
	DrawText(TextFormat("y: %.2f", y), yLinePos.x + 30, yLinePos.y - 10, 12,
			 outputPerceptronColor);
}

void DrawAxis(Machine &machine) {
	static const float thick = 2.5f;
	static const float x = 2000.0f;
	const float		   m = machine.line.m;
	const float		   d = machine.line.d;
	DrawLineEx({0, -x}, {0, x}, thick, GREEN);
	DrawLineEx({-x, 0}, {x, 0}, thick, RED);
	const float yi = calcDeclive(m, -x, d);
	const float yf = calcDeclive(m, x, d);
	DrawLineEx({-x, yi}, {x, yf}, thick, BLUE);
	const size_t index = 0;
	const float	 w0 = machine.brain.getWeightedX0(1, index);
	const float	 w1 = machine.brain.getWeightedX1(1, index);
	const float	 wb = machine.brain.getBias() * machine.brain.getBiasWeight();
	const float	 myi = (-(w0 * -x) - wb) / w1;
	const float	 myf = (-(w0 * x) - wb) / w1;
	DrawLineEx({-x, myi}, {x, myf}, thick, ORANGE);
}

void DrawPoints(Machine &machine) {
	for (size_t i = 0; i < initialPointAmount; i++) {
		const int	  desired = machine.desired[i];
		const Vector2 center = machine.points[i];
		const int	  guess = machine.brain.feedFoward(&center, 1);
		if (desired == 1)
			DrawCircleLinesV(center, pointRadius + 1, pointColor);
		else
			DrawCircleV(center, pointRadius, pointColor);
		if (guess == desired)
			DrawCircleV(center, pointRadius - 1, GREEN);
		else
			DrawCircleV(center, pointRadius - 1, RED);
	}
}

void renderPerceptron(Machine &machine) {
	BeginMode2D(machine.camera);
	ClearBackground(backGroundColor);
	DrawPoints(machine);
	DrawAxis(machine);
	EndMode2D();
}

void renderMainMenu(Machine &machine) {
	static const char	 title[] = "MAIN MENU";
	static const char	 subTitle[] = "PRESS ENTER or SPACE to START";
	static const int	 titleFontSize = windowHeight / 20;
	static const int	 subTitleFontSize = titleFontSize / 2;
	static const int	 titlePosOffset = titleFontSize * sizeof(title) / 3;
	static const Vector2 titlePos =
		(Vector2){screenMiddle.x - titlePosOffset, 20};
	static const int subTitlePosOffset =
		subTitleFontSize * sizeof(subTitle) / 3;
	static const Vector2 subTitlePos =
		(Vector2){screenMiddle.x - subTitlePosOffset, screenMiddle.y};
	ClearBackground(backGroundColor);
	DrawText(title, titlePos.x, titlePos.y, titleFontSize, DARKGREEN);
	DrawText(subTitle, subTitlePos.x, subTitlePos.y, subTitleFontSize,
			 DARKGREEN);
	machine.state = STATE::MAINMENU;
}

int handleNeuralNetworkState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MAINMENU);
	BeginDrawing();
	renderNeuralNetwork(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::NEURALNETWORK);
}

int handlePerceptronState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MAINMENU);
	engineInput(machine);
	BeginDrawing();
	renderPerceptron(machine);
	DrawPerceptron(machine);
	renderImGui(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::PERCEPTRON);
}

int handleMainMenuState(Machine &machine) {
	SetExitKey(KEY_ESCAPE);
	if (IsKeyPressed(KEY_SPACE)) return (STATE::PERCEPTRON);
	if (IsKeyPressed(KEY_ENTER)) return (STATE::NEURALNETWORK);
	BeginDrawing();
	renderMainMenu(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::MAINMENU);
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
