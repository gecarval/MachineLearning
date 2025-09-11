#include "../../includes/Machine.hpp"

// Internal Timer
static float timer = 0;

// Neural Network Image Grid Map
static const float grid = 10;
static const float cols = windowWidth / grid / 2;
static const float rows = windowHeight / grid;

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

int handleNeuralNetworkState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MAINMENU);
	BeginDrawing();
	ClearBackground(backGroundColor);
	renderNeuralNetwork(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::NEURALNETWORK);
}
