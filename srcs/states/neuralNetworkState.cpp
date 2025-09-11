#include "../../includes/Machine.hpp"

// Neural Network Image Grid Map
static const float grid = 10.0f;
static const float cols = windowWidth / grid;
static const float rows = windowHeight / grid / 2;

// Neural Network TrainData
std::vector<std::vector<float>> trainData;
std::vector<std::vector<float>> trainResult;
static const int				trainLen = 2;
static const int				resLen = 1;

static void inputHandler(Machine &machine) {
	if (!IsWindowFocused()) {
		return;
	}
	if (IsKeyPressed(KEY_R)) {
		machine.NN = NeuralNetwork(inputNodes, hiddenNodes, outputNodes,
								   hiddenLayerLength);
		machine.NN.setLearnRate(learningRate);
	}
	if (IsKeyPressed(KEY_D)) {
		trainData.clear();
		trainResult.clear();
	}
	if (IsKeyPressed(KEY_S)) {
		NeuralNetwork::serialize(machine.NN, "NeuralNetwork.json");
	}
	if (IsKeyPressed(KEY_L)) {
		try {
			machine.NN = NeuralNetwork::deserialize("NeuralNetwork.json");
		} catch (const std::runtime_error &error) {
			std::cerr << "[WARNING]" << std::endl;
			std::cerr << error.what() << std::endl;
		}
	}
	if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
		const Vector2 &mousePos = GetMousePosition();
		if (mousePos.y > windowHeight / 2.0f) {
			const float mouseX = Remap(mousePos.x, 0, windowWidth, 0.0f, 1.0f);
			const float mouseY = Remap(mousePos.y, windowHeight / 2.0f,
									   windowHeight, 0.0f, 1.0f);
			std::vector<float> data;
			std::vector<float> res;
			data.push_back(mouseX);
			data.push_back(mouseY);
			res.push_back(1.0f);
			trainData.push_back(data);
			trainResult.push_back(res);
		}
	}
	if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
		const Vector2 &mousePos = GetMousePosition();
		if (mousePos.y > windowHeight / 2.0f) {
			const float mouseX = Remap(mousePos.x, 0, windowWidth, 0.0f, 1.0f);
			const float mouseY = Remap(mousePos.y, windowHeight / 2.0f,
									   windowHeight, 0.0f, 1.0f);
			std::vector<float> data;
			std::vector<float> res;
			data.push_back(mouseX);
			data.push_back(mouseY);
			res.push_back(0.0f);
			trainData.push_back(data);
			trainResult.push_back(res);
		}
	}
}

void DrawNeuralNetwork(const NeuralNetwork &nn, float screenWidth,
					   float screenHeight) {
	// Colors
	Color inputColor = BLUE;
	Color hiddenColor = GREEN;
	Color outputColor = RED;
	Color lineColorPositive = {
		0, 255, 0, 100}; // Semi-transparent green for positive weights
	Color lineColorNegative = {
		255, 0, 0, 100}; // Semi-transparent red for negative weights
	Color textColor = WHITE;

	// Layout parameters
	const float nodeRadius = 20.0f;
	const float layerSpacing =
		screenWidth /
		(nn.getNumberOfHiddenNodes() > 0 ? nn.getHiddenLayerLength() + 2 : 2);
	const float nodeSpacing = 50.0f;
	const float weightMaxThickness = 5.0f;
	const float maxWeight = 2.0f; // For normalizing weight visualization

	// Calculate number of layers
	size_t totalLayers =
		nn.getHiddenLayerLength() + 2; // Input + hidden layers + output
	float xOffset = layerSpacing / 2;

	// Store node positions for drawing connections
	std::vector<std::vector<Vector2>> nodePositions(totalLayers);

	// Draw input layer
	size_t inputNodes = nn.getNumberOfInputsNodes();
	float  yStart = screenHeight / 2 - (inputNodes - 1) * nodeSpacing / 2;
	nodePositions[0].resize(inputNodes);
	for (size_t i = 0; i < inputNodes; ++i) {
		nodePositions[0][i] = {xOffset, yStart + i * nodeSpacing};
		DrawCircleV(nodePositions[0][i], nodeRadius, inputColor);
		DrawText(("I" + std::to_string(i)).c_str(), nodePositions[0][i].x - 10,
				 nodePositions[0][i].y - 5, 15, textColor);
	}
	xOffset += layerSpacing;

	// Draw hidden layers
	for (size_t layer = 0; layer < nn.getHiddenLayerLength(); ++layer) {
		size_t hiddenNodes = nn.getNumberOfHiddenNodes();
		yStart = screenHeight / 2 - (hiddenNodes - 1) * nodeSpacing / 2;
		nodePositions[layer + 1].resize(hiddenNodes);
		for (size_t i = 0; i < hiddenNodes; ++i) {
			nodePositions[layer + 1][i] = {xOffset, yStart + i * nodeSpacing};
			DrawCircleV(nodePositions[layer + 1][i], nodeRadius, hiddenColor);
			DrawText(
				("H" + std::to_string(layer) + "_" + std::to_string(i)).c_str(),
				nodePositions[layer + 1][i].x - 15,
				nodePositions[layer + 1][i].y - 5, 15, textColor);
		}
		xOffset += layerSpacing;
	}

	// Draw output layer
	size_t outputNodes = nn.getNumberOfOutputsNodes();
	yStart = screenHeight / 2 - (outputNodes - 1) * nodeSpacing / 2;
	nodePositions[totalLayers - 1].resize(outputNodes);
	for (size_t i = 0; i < outputNodes; ++i) {
		nodePositions[totalLayers - 1][i] = {xOffset, yStart + i * nodeSpacing};
		DrawCircleV(nodePositions[totalLayers - 1][i], nodeRadius, outputColor);
		DrawText(("O" + std::to_string(i)).c_str(),
				 nodePositions[totalLayers - 1][i].x - 10,
				 nodePositions[totalLayers - 1][i].y - 5, 15, textColor);
	}

	// Draw weights (lines between layers)
	for (size_t layer = 0; layer < nn.getHiddenLayerLength() + 1; ++layer) {
		const DMatrix &weights = nn.getWeigthAt(layer);
		for (size_t i = 0; i < weights.getRowLength(); ++i) {
			for (size_t j = 0; j < weights.getColLength(); ++j) {
				Vector2 start = nodePositions[layer][j];
				Vector2 end = nodePositions[layer + 1][i];
				float	weight = weights.getValue(i, j);
				float	thickness =
					std::min(weightMaxThickness,
							 std::abs(weight) / maxWeight * weightMaxThickness);
				Color lineColor =
					weight >= 0 ? lineColorPositive : lineColorNegative;
				DrawLineEx(start, end, thickness, lineColor);
			}
		}
	}

	// Draw biases (as small circles or labels near nodes)
	for (size_t layer = 0; layer < nn.getHiddenLayerLength() + 1; ++layer) {
		const DMatrix &biases = nn.getBiasAt(layer);
		for (size_t i = 0; i < biases.getRowLength(); ++i) {
			Vector2 nodePos = nodePositions[layer + 1][i];
			float	bias = biases.getValue(i, 0);
			DrawCircleV({nodePos.x + nodeRadius + 10, nodePos.y}, 5.0f, YELLOW);
			DrawText(TextFormat("%.2f", bias), nodePos.x + nodeRadius + 15,
					 nodePos.y - 5, 10, textColor);
		}
	}
}

void trainMachineNeuralNetwork(Machine &machine) {
	const int				  samples = trainData.size();
	const int				  randomIndex = GetRandomValue(0, samples - 1);
	const std::vector<float> &train(trainData[randomIndex]);
	const std::vector<float> &res(trainResult[randomIndex]);
	machine.NN.train(train, res);
}

void renderNeuralNetwork(Machine &machine) {
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			const float				 x0 = i / cols;
			const float				 x1 = j / rows;
			const float				 x = i * grid;
			const float				 y = j * grid + windowHeight / 2.0f;
			const Vector2			 gridPos = (Vector2){x, y};
			const Vector2			 gridSize = (Vector2){grid, grid};
			const std::vector<float> input = {x0, x1};
			const std::vector<float> output = machine.NN.feedFoward(input);
			const unsigned char		 alpha = Remap(output[0], 0, 1, 0, 255);
			const Color gridColor = (Color){alpha, alpha, alpha, 255};
			DrawRectangleV(gridPos, gridSize, gridColor);
		}
	}
	for (size_t i = 0; i < trainData.size(); i++) {
		static const float radius = 2.0f;
		static const Color color1 = RED;
		static const Color color2 = GREEN;
		const float pointX = Remap(trainData[i][0], 0.0f, 1.0f, 0, windowWidth);
		const float pointY = Remap(trainData[i][1], 0.0f, 1.0f,
								   windowHeight / 2.0f, windowHeight);
		const Vector2 center = {pointX, pointY};
		if (trainResult[i][0] > 0.0f) {
			DrawCircleV(center, radius, color1);
		} else {
			DrawCircleV(center, radius, color2);
		}
	}
}

int handleNeuralNetworkState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MAINMENU);
	inputHandler(machine);
	static const int trainLoop = 1000;
	if (!trainData.empty()) {
		for (int i = 0; i < trainLoop; i++) {
			trainMachineNeuralNetwork(machine);
		}
	}
	BeginDrawing();
	ClearBackground(BLACK);
	renderNeuralNetwork(machine);
	DrawNeuralNetwork(machine.NN, windowWidth, windowHeight / 2.0f);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::NEURALNETWORK);
}
