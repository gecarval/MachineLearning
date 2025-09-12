#include "../../includes/Machine.hpp"

// Neural Network TrainData
std::vector<std::vector<float>> trainData;
std::vector<std::vector<float>> trainResult;
static const int				trainLen = 2;
static const int				resLen = 1;

static void inputHandler(Machine &machine) {
	if (!IsWindowFocused()) {
		return;
	}
	if (IsKeyDown(KEY_LEFT_CONTROL)) {
		if (IsKeyPressed(KEY_Z)) {
			if (!trainData.empty()) {
				trainData.pop_back();
				trainResult.pop_back();
			}
		}
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
		if (mousePos.y > GetScreenHeight() / 2.0f) {
			const float mouseX =
				Remap(mousePos.x, 0, GetScreenWidth(), 0.0f, 1.0f);
			const float mouseY = Remap(mousePos.y, GetScreenHeight() / 2.0f,
									   GetScreenHeight(), 0.0f, 1.0f);
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
		if (mousePos.y > GetScreenHeight() / 2.0f) {
			const float mouseX =
				Remap(mousePos.x, 0, GetScreenWidth(), 0.0f, 1.0f);
			const float mouseY = Remap(mousePos.y, GetScreenHeight() / 2.0f,
									   GetScreenHeight(), 0.0f, 1.0f);
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
	static const Color inputColor = BLUE;
	static const Color hiddenColor = GREEN;
	static const Color outputColor = RED;
	static const Color lineColorPositive = {
		0, 255, 0, 100}; // Semi-transparent green for positive weights
	static const Color lineColorNegative = {
		255, 0, 0, 100}; // Semi-transparent red for negative weights
	static const Color textColor = WHITE;
	static const Color biasColor = YELLOW;

	// Calculate number of layers and max nodes for dynamic scaling
	size_t totalLayers =
		nn.getHiddenLayerLength() + 2; // Input + hidden layers + output
	size_t inputNodes = nn.getNumberOfInputsNodes();
	size_t hiddenNodes = nn.getNumberOfHiddenNodes();
	size_t outputNodes = nn.getNumberOfOutputsNodes();
	size_t maxNodesPerLayer = std::max({inputNodes, hiddenNodes, outputNodes});

	// Layout parameters (fully dynamic based on layers and nodes)
	const float usableHeight = screenHeight * 0.9f;
	const float usableWidth = screenWidth * 0.9f;
	const float nodeSpacing = (maxNodesPerLayer > 1)
								  ? usableHeight / (maxNodesPerLayer - 1)
								  : usableHeight * 0.5f;
	const float layerSpacing = (totalLayers > 1)
								   ? usableWidth / (totalLayers - 1)
								   : usableWidth * 0.5f;
	const float nodeRadius = std::min(screenWidth, screenHeight) * 0.015f;
	const float weightMaxThickness = nodeRadius * 0.10f;
	const float weightMinThickness = nodeRadius * 0.02f;
	const float maxWeight = 2.0f;
	const float textSize = nodeRadius * 1.5f;
	const float biasTextSize = textSize * 0.75f;
	const float biasCircleRadius = nodeRadius * 0.25f;

	// Remap function for weight thickness (logarithmic scaling)
	auto remapWeightToThickness = [&](float weight) -> float {
		float absWeight = std::abs(weight);
		// Logarithmic scaling: thickness = min + (max - min) * log(1 +
		// absWeight) / log(1 + maxWeight)
		float normalized =
			std::log1p(absWeight) /
			std::log1p(maxWeight); // log1p for stability with small values
		return weightMinThickness +
			   (weightMaxThickness - weightMinThickness) * normalized;
	};

	// Initial xOffset for first layer
	float xOffset = layerSpacing * 0.2f;

	// Store node positions for drawing connections
	std::vector<std::vector<Vector2>> nodePositions(totalLayers);

	// Draw input layer
	float yStart = screenHeight * 0.5f - (inputNodes - 1) * nodeSpacing * 0.5f;
	nodePositions[0].resize(inputNodes);
	for (size_t i = 0; i < inputNodes; ++i) {
		nodePositions[0][i] = {xOffset, yStart + i * nodeSpacing};
		DrawCircleV(nodePositions[0][i], nodeRadius, inputColor);
	}
	xOffset += layerSpacing;

	// Draw hidden layers (assuming uniform hidden layer size)
	for (size_t layer = 0; layer < nn.getHiddenLayerLength(); ++layer) {
		yStart = screenHeight * 0.5f - (hiddenNodes - 1) * nodeSpacing * 0.5f;
		nodePositions[layer + 1].resize(hiddenNodes);
		for (size_t i = 0; i < hiddenNodes; ++i) {
			nodePositions[layer + 1][i] = {xOffset, yStart + i * nodeSpacing};
			DrawCircleV(nodePositions[layer + 1][i], nodeRadius, hiddenColor);
		}
		xOffset += layerSpacing;
	}

	// Draw output layer
	yStart = screenHeight * 0.5f - (outputNodes - 1) * nodeSpacing * 0.5f;
	nodePositions[totalLayers - 1].resize(outputNodes);
	for (size_t i = 0; i < outputNodes; ++i) {
		nodePositions[totalLayers - 1][i] = {xOffset, yStart + i * nodeSpacing};
		DrawCircleV(nodePositions[totalLayers - 1][i], nodeRadius, outputColor);
	}

	// Draw weights (lines between layers)
	for (size_t layer = 0; layer < nn.getHiddenLayerLength() + 1; ++layer) {
		const DMatrix &weights = nn.getWeigthAt(layer);
		for (size_t i = 0; i < weights.getRowLength(); ++i) {
			for (size_t j = 0; j < weights.getColLength(); ++j) {
				Vector2 start = nodePositions[layer][j];
				Vector2 end = nodePositions[layer + 1][i];
				float	weight = weights.getValue(i, j);
				float	thickness = remapWeightToThickness(weight);
				Color	lineColor =
					  weight >= 0 ? lineColorPositive : lineColorNegative;
				DrawLineEx(start, end, thickness, lineColor);
			}
		}
	}

	// Draw biases (as small circles and labels near nodes)
	for (size_t layer = 0; layer < nn.getHiddenLayerLength() + 1; ++layer) {
		const DMatrix &biases = nn.getBiasAt(layer);
		for (size_t i = 0; i < biases.getRowLength(); ++i) {
			Vector2 nodePos = nodePositions[layer + 1][i];
			float	bias = biases.getValue(i, 0);
			DrawCircleV({nodePos.x + nodeRadius + nodeRadius * 0.5f, nodePos.y},
						biasCircleRadius, biasColor);
			DrawText(TextFormat("%.2f", bias),
					 nodePos.x + nodeRadius + nodeRadius * 0.75f,
					 nodePos.y - nodeRadius * 0.25f, biasTextSize, textColor);
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
	// Neural Network Image Grid Map
	static const float amount = 50.0f;
	const Vector2	   grid =
		(Vector2){GetScreenWidth() / amount, GetScreenHeight() / amount};
	const float cols = GetScreenWidth() / grid.x;
	const float rows = (GetScreenHeight() / 2.0f) / grid.y;

	for (int i = 0; i <= rows + 1; i++) {
		for (int j = 0; j < cols; j++) {
			const float				 x0 = j / cols;
			const float				 x1 = i / rows;
			const float				 x = j * grid.x;
			const float				 y = i * grid.y + GetScreenHeight() / 2.0f;
			const Vector2			 gridPos = (Vector2){x, y};
			const std::vector<float> input = {x0, x1};
			const std::vector<float> output = machine.NN.feedFoward(input);
			const unsigned char		 alpha = Remap(output[0], 0, 1, 0, 255);
			const Color gridColor = (Color){alpha, alpha, alpha, 255};
			DrawRectangleV(gridPos, grid, gridColor);
		}
	}
	for (size_t i = 0; i < trainData.size(); i++) {
		static const float radius = 2.0f;
		static const Color color1 = RED;
		static const Color color2 = GREEN;
		const float		   pointX =
			Remap(trainData[i][0], 0.0f, 1.0f, 0, GetScreenWidth());
		const float	  pointY = Remap(trainData[i][1], 0.0f, 1.0f,
									 GetScreenHeight() / 2.0f, GetScreenHeight());
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
	static const int trainLoop = 100;
	if (!trainData.empty()) {
		for (int i = 0; i < trainLoop; i++) {
			trainMachineNeuralNetwork(machine);
		}
	}
	BeginDrawing();
	ClearBackground(BLACK);
	renderNeuralNetwork(machine);
	DrawNeuralNetwork(machine.NN, GetScreenWidth(), GetScreenHeight() / 2.0f);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::NEURALNETWORK);
}
