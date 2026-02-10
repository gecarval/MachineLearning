#include "../../includes/Machine.hpp"

// Neural Network Image Grid Map
static RenderTexture2D tmp;
static bool			   isTmpLoaded = false;
static const float	   factor = 1.0f;
static const float	   size = 25.0f * factor;
static const float	   imageScale = 15.0f / factor;

// Neural Network TrainData
Camera3D camera = {(Vector3){12.0f, 17.0f, 12.0f}, (Vector3){0.0f, 7.0f, 0.0f},
				   (Vector3){0.0f, 1.0f, 0.0f}, 45.0f, CAMERA_PERSPECTIVE};
std::vector<std::vector<float>> trainData;
std::vector<std::vector<float>> trainResult;
static const int				trainLen = 2;
static const int				resLen = 1;

static void onMouseClick(const unsigned int buttonCode) {
	const Vector2 &mousePos = GetMousePosition();
	const Vector2  padding = {0, (GetScreenHeight() / 2.0f)};
	const Vector2  paddingEnd =
		(Vector2){padding.x + (tmp.texture.width * imageScale),
				  padding.y + (tmp.texture.height * imageScale)};
	if (mousePos.x > padding.x && mousePos.x < paddingEnd.x &&
		mousePos.y > padding.y && mousePos.y < paddingEnd.y) {
		const float mouseX =
			Remap(mousePos.x, padding.x, paddingEnd.x, 0.0f, 1.0f);
		const float mouseY =
			Remap(mousePos.y, padding.y, paddingEnd.y, 0.0f, 1.0f);
		std::vector<float> data;
		std::vector<float> res;
		data.push_back(mouseX);
		data.push_back(mouseY);
		if (MOUSE_BUTTON_LEFT == buttonCode) {
			res.push_back(1.0f);
		} else {
			res.push_back(0.0f);
		}
		trainData.push_back(data);
		trainResult.push_back(res);
	}
}

static void inputHandler(Machine &machine) {
	if (!IsWindowFocused()) {
		return;
	}
	if (IsKeyDown(KEY_LEFT_CONTROL)) {
		if (IsKeyPressed(KEY_Z)) {
			if (!trainData.empty() && !trainResult.empty()) {
				trainData.pop_back();
				trainResult.pop_back();
			}
		}
	}
	if (IsKeyPressed(KEY_UP)) {
		machine.NN.setLearnRate(machine.NN.getLearnRate() * 10.0f);
	}
	if (IsKeyPressed(KEY_DOWN)) {
		machine.NN.setLearnRate(machine.NN.getLearnRate() / 10.0f);
	}
	if (IsKeyPressed(KEY_R)) {
		const float ln = machine.NN.getLearnRate();
		machine.NN = NeuralNetwork(machine.NN.getNumberOfInputsNodes(),
								   machine.NN.getNumberOfHiddenNodes(),
								   machine.NN.getNumberOfOutputsNodes(),
								   machine.NN.getHiddenLayerLength());
		machine.NN.setLearnRate(ln);
		machine.NN.enableSoftmax(true);
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
	if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
		onMouseClick(MOUSE_BUTTON_LEFT);
	}
	if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
		onMouseClick(MOUSE_BUTTON_RIGHT);
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
		const DMatrix &weights = nn.getWeightAt(layer);
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

void renderMap(Machine &machine) {
	if (!isTmpLoaded) {
		tmp = LoadRenderTexture(size, size);
		isTmpLoaded = true;
	}
	BeginTextureMode(tmp);
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			const std::vector<float> input = {Remap(x, 0, size, 0, 1.0f),
											  Remap(y, 0, size, 1.0f, 0)};
			const std::vector<float> output = machine.NN.feedForward(input);
			const unsigned char		 alpha = Remap(output[0], 0, 1, 0, 255);
			const Color gridColor = (Color){alpha, alpha, alpha, 255};
			DrawPixel(x, y, gridColor);
		}
	}
	EndTextureMode();
}

void renderPoints(Machine &machine) {
	const Vector2 padding = {0, (GetScreenHeight() / 2.0f)};
	const Vector2 paddingEnd =
		(Vector2){padding.x + (tmp.texture.width * imageScale),
				  padding.y + (tmp.texture.height * imageScale)};
	static const Vector3 mapPos = (Vector3){-2, 0, -2};
	static const Vector3 cubPos = (Vector3){0, 2, 0};
	static const Vector3 mapSize = (Vector3){4, 4, 4};
	const Image			 mapImage = LoadImageFromTexture(tmp.texture);
	const Mesh			 mapMesh = GenMeshHeightmap(mapImage, mapSize);
	const Model			 mapModel = LoadModelFromMesh(mapMesh);

	mapModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = tmp.texture;
	BeginMode3D(camera);
	DrawModel(mapModel, mapPos, 1.0f, BLUE);
	DrawGrid(5, 1.0f);
	DrawCubeWires(cubPos, 4, 4, 4, RED);
	EndMode3D();
	DrawTextureEx(tmp.texture, (Vector2){0, GetScreenHeight() / 2.0f}, 0,
				  imageScale, WHITE);

	static const float radius = 2.0f;
	static const Color color1 = RED;
	static const Color color2 = GREEN;
	for (size_t i = 0; i < trainData.size(); i++) {
		const float pointX =
			Remap(trainData[i][0], 0.0f, 1.0f, padding.x, paddingEnd.x);
		const float pointY =
			Remap(trainData[i][1], 0.0f, 1.0f, padding.y, paddingEnd.y);
		const Vector2 center = {pointX, pointY};
		if (trainResult[i][0] > 0.0f) {
			DrawCircleV(center, radius, color1);
		} else {
			DrawCircleV(center, radius, color2);
		}
	}
	DrawText(TextFormat("learnRate: %5.10f", machine.NN.getLearnRate()), 20, 50,
			 10, WHITE);
	UnloadImage(mapImage);
	UnloadModel(mapModel);
}

int handleNeuralNetworkState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MENU::MAIN);
	inputHandler(machine);
	static const int trainLoop = 100;
	if (!trainData.empty()) {
		for (int i = 0; i < trainLoop; i++) {
			trainMachineNeuralNetwork(machine);
		}
	}
	UpdateCamera(&camera, CAMERA_ORBITAL);
	renderMap(machine);
	BeginDrawing();
	ClearBackground(BLACK);
	renderPoints(machine);
	DrawNeuralNetwork(machine.NN, GetScreenWidth(), GetScreenHeight() / 2.0f);
	backButton(machine).draw();
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (machine.state);
}
