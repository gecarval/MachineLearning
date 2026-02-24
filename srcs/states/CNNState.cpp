#include "../../includes/Machine.hpp"

// Canvas configuration
namespace CanvasConfig {
constexpr int GRID_SIZE = 28;
// Brush radius in grid cells with Gaussian falloff.
// Matches closely with MNIST Database
constexpr int BRUSH_RADIUS = 1;
// Gaussian sigma for brush softness — mimics ink diffusion in MNIST samples.
constexpr float BRUSH_SIGMA = 0.60f;

// Dynamic sizing based on screen
inline float getCellSize() {
	const float minDimension = std::min(GetScreenWidth(), GetScreenHeight());
	return (std::max(10.0f, minDimension / 40.0f));
}

inline float getCanvasSize() {
	return (GRID_SIZE * getCellSize());
}
} // namespace CanvasConfig

struct Canvas {
	std::vector<std::vector<float>> grid;
	Vector2							position;

	Canvas()
		: grid(CanvasConfig::GRID_SIZE,
			   std::vector<float>(CanvasConfig::GRID_SIZE, 0.0f)),
		  position{0, 0} {
	}

	bool importImageCanvas(const std::string &filename) {
		Image canvasImage{};
		canvasImage = LoadImage(filename.c_str());
		if (canvasImage.data) {
			TraceLog(LOG_INFO, "Canvas imported successfully from %s",
					 filename.c_str());
		} else {
			TraceLog(LOG_ERROR, "Failed to import canvas image!");
			return (true);
		}
		for (int y = 0; y < canvasImage.height; y++) {
			for (int x = 0; x < canvasImage.width; x++) {
				const Color colorVal = GetImageColor(canvasImage, x, y);
				const float value = static_cast<float>(colorVal.r) / 255.0f;
				this->grid[y][x] = value;
			}
		}
		UnloadImage(canvasImage);
		return (false);
	}

	void exportImageCanvas(const std::string &filename) const {
		const int width = CanvasConfig::GRID_SIZE;
		const int height = CanvasConfig::GRID_SIZE;
		Image	  canvasImage = GenImageColor(width, height, BLACK);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				const float			val = this->grid[y][x];
				const unsigned char colorVal =
					static_cast<unsigned char>(val * 255.0f);
				const Color pixelColor =
					(Color){colorVal, colorVal, colorVal, 255};
				ImageDrawPixel(&canvasImage, x, y, pixelColor);
			}
		}
		if (ExportImage(canvasImage, filename.c_str())) {
			TraceLog(LOG_INFO, "Canvas exported successfully to %s",
					 filename.c_str());
		} else {
			TraceLog(LOG_ERROR, "Failed to export canvas image!");
		}
		UnloadImage(canvasImage);
	}

	void putConvolutionResultOnCanvas(Machine			&machine,
									  std::vector<float> input,
									  const size_t filter, const size_t layer) {
		// Build a column DMatrix from the flat input vector as
		// getConvOnAFilterFunnel expects.
		DMatrix inputMatrix(input.size(), 1);
		for (size_t i = 0; i < input.size(); ++i) {
			inputMatrix(i, 0) = input[i];
		}
		DMatrix featureMap =
			machine.CNN.getConvOnAFilterFunnel(inputMatrix, filter, layer);
		featureMap = featureMap.transpose();
		for (size_t r = 0; r < featureMap.getRowLength(); r++) {
			for (size_t c = 0; c < featureMap.getColLength(); c++) {
				this->setPixel(static_cast<int>(r), static_cast<int>(c),
							   featureMap(r, c));
			}
		}
	}

	void clear() {
		for (auto &row : this->grid) {
			std::fill(row.begin(), row.end(), 0.0f);
		}
	}

	// Gaussian brush: accumulates ink with a smooth falloff so stroke edges
	// produced by anti-aliased pen tablets
	void paintPixelGaussian(int cx, int cy, float value) {
		const int	R = CanvasConfig::BRUSH_RADIUS;
		const float sigma = CanvasConfig::BRUSH_SIGMA;
		const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
		for (int dy = -R; dy <= R; ++dy) {
			for (int dx = -R; dx <= R; ++dx) {
				const float dist2 = static_cast<float>(dx * dx + dy * dy);
				const float weight = std::exp(-dist2 * inv2s2);
				const int	px = cx + dx;
				const int	py = cy + dy;
				if (px >= 0 && px < CanvasConfig::GRID_SIZE && py >= 0 &&
					py < CanvasConfig::GRID_SIZE) {
					// Additive blend, clamped to [0,1] — lets strokes overlap
					// and build up, the same way real ink does.
					float &cell = this->grid[py][px];
					cell = std::min(1.0f, cell + value * weight);
				}
			}
		}
	}

	// Re-centres the drawn digit inside the 28×28 grid using centre-of-mass,
	// matching the MNIST normalisation pipeline (Lecun et al., 1998).
	// Leaves a 2-pixel margin on all sides, consistent with MNIST padding.
	void centreDigit() {
		constexpr int G = CanvasConfig::GRID_SIZE;

		// 1. Compute bounding box of non-zero pixels.
		int minR = G, maxR = -1, minC = G, maxC = -1;
		for (int r = 0; r < G; ++r) {
			for (int c = 0; c < G; ++c) {
				if (this->grid[r][c] > 0.01f) {
					if (r < minR) minR = r;
					if (r > maxR) maxR = r;
					if (c < minC) minC = c;
					if (c > maxC) maxC = c;
				}
			}
		}
		// canvas is empty
		if (maxR < 0) {
			return;
		}

		// 2. Compute centre of mass (weighted by pixel value).
		float sumW = 0.0f, sumR = 0.0f, sumC = 0.0f;
		for (int r = minR; r <= maxR; ++r) {
			for (int c = minC; c <= maxC; ++c) {
				const float w = this->grid[r][c];
				sumW += w;
				sumR += w * static_cast<float>(r);
				sumC += w * static_cast<float>(c);
			}
		}
		const float comR = (sumW > 0.0f) ? sumR / sumW : (minR + maxR) / 2.0f;
		const float comC = (sumW > 0.0f) ? sumC / sumW : (minC + maxC) / 2.0f;

		// 3. Compute integer shift to move centre of mass to the canvas centre.
		const float targetR = (G - 1) / 2.0f;
		const float targetC = (G - 1) / 2.0f;
		const int	shiftR = static_cast<int>(std::round(targetR - comR));
		const int	shiftC = static_cast<int>(std::round(targetC - comC));

		if (shiftR == 0 && shiftC == 0) {
			return;
		}

		// 4. Copy with the shift applied (unvisited cells stay 0).
		std::vector<std::vector<float>> shifted(G, std::vector<float>(G, 0.0f));
		for (int r = 0; r < G; ++r) {
			for (int c = 0; c < G; ++c) {
				const int nr = r + shiftR;
				const int nc = c + shiftC;
				if (nr >= 0 && nr < G && nc >= 0 && nc < G)
					shifted[nr][nc] = this->grid[r][c];
			}
		}
		this->grid = std::move(shifted);
	}

	// Hard-set a single pixel — used for erasing and image import.
	void setPixel(int x, int y, float value) {
		if (x >= 0 && x < CanvasConfig::GRID_SIZE && y >= 0 &&
			y < CanvasConfig::GRID_SIZE) {
			this->grid[y][x] = Clamp(value, 0.0f, 1.0f);
		}
	}

	float getPixel(int x, int y) const {
		if (x >= 0 && x < CanvasConfig::GRID_SIZE && y >= 0 &&
			y < CanvasConfig::GRID_SIZE) {
			return (this->grid[y][x]);
		}
		return (0);
	}

	std::vector<float> toVector() const {
		std::vector<float> result;
		result.reserve(CanvasConfig::GRID_SIZE * CanvasConfig::GRID_SIZE);
		for (const auto &row : this->grid) {
			result.insert(result.end(), row.begin(), row.end());
		}
		return (result);
	}
};

// UI state
struct CNNState {
	Canvas			   canvas;
	int				   currentLabel = 0;
	bool			   isPredicting = false;
	std::vector<float> predictions;

	CNNState() : predictions(10, 0.0f) {
	}
};

static CNNState cnnState;

// Layout calculation
struct Layout {
	Vector2 canvasPos;
	Vector2 predictionPos;
	Vector2 instructionPos;
	float	titleY;
	float	cellSize;
	float	canvasSize;
	bool	showPredictionsSide;

	static Layout calculate() {
		Layout layout;

		const int screenW = GetScreenWidth();
		const int screenH = GetScreenHeight();

		layout.cellSize = CanvasConfig::getCellSize();
		layout.canvasSize = CanvasConfig::getCanvasSize();
		layout.titleY = 20.0f;

		// Determine if we have space for side-by-side layout
		constexpr float PREDICTION_WIDTH = 300.0f;
		constexpr float MIN_PADDING = 20.0f;
		layout.showPredictionsSide =
			(screenW >= layout.canvasSize + PREDICTION_WIDTH + MIN_PADDING * 3);

		// Calculate canvas position (centered or left-aligned)
		if (layout.showPredictionsSide) {
			// Side-by-side layout
			const float totalWidth =
				layout.canvasSize + PREDICTION_WIDTH + MIN_PADDING;
			layout.canvasPos.x = (screenW - totalWidth) / 2.0f;
			layout.predictionPos.x =
				layout.canvasPos.x + layout.canvasSize + MIN_PADDING;
		} else {
			// Stacked layout
			layout.canvasPos.x = (screenW - layout.canvasSize) / 2.0f;
			layout.predictionPos.x = MIN_PADDING;
		}

		// Vertical positioning
		const float titleHeight = 60.0f;
		const float instructionHeight = 120.0f;
		const float availableHeight = screenH - titleHeight - instructionHeight;

		if (layout.showPredictionsSide) {
			// Center canvas vertically
			layout.canvasPos.y =
				titleHeight + (availableHeight - layout.canvasSize) / 2.0f;
			layout.predictionPos.y = layout.canvasPos.y;
		} else {
			// Stack vertically with predictions below
			layout.canvasPos.y = titleHeight + 20.0f;
			layout.predictionPos.y =
				layout.canvasPos.y + layout.canvasSize + 20.0f;
		}

		// Instructions at bottom
		layout.instructionPos.x = 20.0f;
		layout.instructionPos.y = screenH - instructionHeight + 10.0f;

		return (layout);
	}
};

// Helper functions
static void drawCanvas(const Canvas &canvas, float cellSize) {
	const Vector2 &pos = canvas.position;
	const float	   canvasSize = cellSize * CanvasConfig::GRID_SIZE;

	// Draw grid
	for (int y = 0; y < CanvasConfig::GRID_SIZE; ++y) {
		for (int x = 0; x < CanvasConfig::GRID_SIZE; ++x) {
			const float			value = canvas.getPixel(x, y);
			const unsigned char brightness =
				static_cast<unsigned char>(value * 255);
			const Color cellColor = {brightness, brightness, brightness, 255};

			const Rectangle cell = {pos.x + x * cellSize, pos.y + y * cellSize,
									cellSize, cellSize};

			DrawRectangleRec(cell, cellColor);

			// Only draw grid lines if cells are large enough
			if (cellSize >= 15.0f) {
				DrawRectangleLinesEx(cell, 1.0f, LIGHTGRAY);
			}
		}
	}

	// Draw border
	DrawRectangleLinesEx({pos.x, pos.y, canvasSize, canvasSize}, 2.0f, BLACK);
}

static void drawPredictions(const std::vector<float> &predictions,
							Vector2 position, bool compact) {
	const float BAR_WIDTH = compact ? 150.0f : 200.0f;
	const float BAR_HEIGHT = compact ? 25.0f : 30.0f;
	const float SPACING = compact ? 3.0f : 5.0f;
	const int	titleSize = compact ? 18 : 20;
	const int	labelSize = compact ? 16 : 18;
	const int	percentSize = compact ? 14 : 16;

	DrawText("Predictions:", position.x, position.y - 30, titleSize, BLACK);

	// Find max prediction for highlighting
	int	  maxIdx = 0;
	float maxVal = predictions[0];
	for (int i = 1; i < 10; ++i) {
		if (predictions[i] > maxVal) {
			maxVal = predictions[i];
			maxIdx = i;
		}
	}

	for (int i = 0; i < 10; ++i) {
		const float yPos = position.y + i * (BAR_HEIGHT + SPACING);

		// Draw label
		DrawText(TextFormat("%d:", i), position.x, yPos + 6, labelSize, BLACK);

		// Draw background bar
		const Rectangle bgBar = {position.x + 25, yPos, BAR_WIDTH, BAR_HEIGHT};
		DrawRectangleRec(bgBar, LIGHTGRAY);

		// Draw prediction bar
		const float		confidence = Clamp(predictions[i], 0.0f, 1.0f);
		const Rectangle predBar = {position.x + 25, yPos,
								   BAR_WIDTH * confidence, BAR_HEIGHT};

		// Color based on if it's the current label or highest prediction
		Color barColor = BLUE;
		if (i == cnnState.currentLabel) {
			barColor = GREEN;
		} else if (i == maxIdx && cnnState.isPredicting) {
			barColor = ORANGE;
		}
		DrawRectangleRec(predBar, barColor);

		// Draw border
		DrawRectangleLinesEx(bgBar, 1.0f, BLACK);

		// Draw percentage
		const float percentX = position.x + 30 + BAR_WIDTH;
		DrawText(TextFormat("%.1f%%", confidence * 100), percentX, yPos + 6,
				 percentSize, BLACK);
	}
}

static void handleCanvasInput(Canvas &canvas, float cellSize) {
	if (!IsWindowFocused()) {
		return;
	}

	const Vector2	mousePos = GetMousePosition();
	const float		canvasSize = cellSize * CanvasConfig::GRID_SIZE;
	const Rectangle canvasBounds = {canvas.position.x, canvas.position.y,
									canvasSize, canvasSize};

	if (CheckCollisionPointRec(mousePos, canvasBounds)) {
		const int gridX =
			static_cast<int>((mousePos.x - canvas.position.x) / cellSize);
		const int gridY =
			static_cast<int>((mousePos.y - canvas.position.y) / cellSize);

		// Draw with Gaussian brush — produces soft, anti-aliased strokes
		// that match the ink-diffusion characteristics of MNIST samples.
		if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
			canvas.paintPixelGaussian(gridX, gridY, 1.0f);
		}

		// Erase with a hard square brush (right mouse button).
		if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
			const int R = CanvasConfig::BRUSH_RADIUS;
			for (int dy = -R; dy <= R; ++dy) {
				for (int dx = -R; dx <= R; ++dx) {
					canvas.setPixel(gridX + dx, gridY + dy, 0.0f);
				}
			}
		}
	}
}

void loadTrainingData(
	std::vector<std::pair<std::vector<float>, std::vector<float>>> &data) {
	const std::string directory = "traindata/";
	for (int number = 0; number < 10; number++) {
		for (int fileNumber = 0; fileNumber < 10000; fileNumber++) {
			const std::string subdirectory = std::to_string(number);
			const std::string num = std::to_string(fileNumber);
			const std::string pad = std::string(5 - num.length(), '0');
			const std::string image = "/image_" + pad + num;
			const std::string filename = directory + subdirectory + image;
			const std::string extension = ".png";
			if (!std::filesystem::exists(filename + extension)) {
				continue;
			}
			Canvas tempCanvas;
			if (!tempCanvas.importImageCanvas(filename + extension)) {
				std::vector<float> input = tempCanvas.toVector();
				std::vector<float> target(10, 0.0f);
				target[number] = 1.0f;
				data.push_back(std::make_pair(input, target));
			}
		}
	}
}

void DrawLoadingBar(int currentEpoch, int totalEpochs, int currentIndex,
					int totalItems, double averageItemTime) {
	const float progress =
		static_cast<float>(currentIndex) / static_cast<float>(totalItems);
	const int barWidth = 400;
	const int barHeight = 30;
	const int x = (GetScreenWidth() - barWidth) / 2;
	const int y = (GetScreenHeight() - barHeight) / 2;

	// 1. Items left in the CURRENT epoch
	const int	 itemsLeftInCurrentEpoch = totalItems - currentIndex;
	const double timeLeftThisEpoch = averageItemTime * itemsLeftInCurrentEpoch;

	// 2. Full epochs remaining AFTER the current one
	const int	 fullEpochsRemaining = totalEpochs - currentEpoch - 1;
	const double timeForFutureEpochs =
		averageItemTime * totalItems * fullEpochsRemaining;

	const double timeLeftForAllTraining =
		timeLeftThisEpoch + timeForFutureEpochs;

	// Formatting seconds into MM:SS for better readability
	const int mins = static_cast<int>(timeLeftForAllTraining) / 60;
	const int secs = static_cast<int>(timeLeftForAllTraining) % 60;

	ClearBackground(RAYWHITE);

	DrawText(
		TextFormat("Training... Epoch %d/%d", currentEpoch + 1, totalEpochs), x,
		y - 40, 20, BLACK);

	DrawText(TextFormat("Estimated time left: %02d:%02d", mins, secs), x,
			 y + 40, 20, DARKGRAY);

	// Progress bar
	DrawRectangle(x, y, barWidth, barHeight, LIGHTGRAY);
	DrawRectangle(x, y, static_cast<int>(barWidth * progress), barHeight, BLUE);
	DrawRectangleLinesEx(
		{(float)x, (float)y, (float)barWidth, (float)barHeight}, 2.0f, BLACK);
}

void trainModel(Machine &machine, const int epoch) {
	std::vector<std::pair<std::vector<float>, std::vector<float>>> trainingData;
	loadTrainingData(trainingData);
	if (trainingData.empty()) {
		return;
	}
	// Randomize training data order for better training
	std::shuffle(trainingData.begin(), trainingData.end(),
				 std::default_random_engine(std::random_device{}()));
	Button cancelTrainButton(GetScreenWidth() / 2.0f - 50,
							 GetScreenHeight() / 2.0f + 80, 100, 30, "Cancel");
	double averageItemTime = 0.0;
	long   totalItemsProcessed = 0;
	for (int e = 0; e < epoch; ++e) {
		for (long i = 0; i < static_cast<long>(trainingData.size()); ++i) {
			double startTime = GetTime();
			machine.CNN.train(trainingData[i].first, trainingData[i].second);
			// Cumulative Moving Average for smoother time estimation
			double frameTime = GetTime() - startTime;
			totalItemsProcessed++;
			averageItemTime =
				(averageItemTime * (totalItemsProcessed - 1) + frameTime) /
				totalItemsProcessed;
			// draw every X items to improve performance (Optional)
			if (i % 10 == 0) {
				BeginDrawing();
				DrawLoadingBar(e, epoch, i, trainingData.size(),
							   averageItemTime);

				cancelTrainButton.update();
				cancelTrainButton.draw();

				if (cancelTrainButton.isButtonPressed()) {
					EndDrawing();
					TraceLog(LOG_INFO, "Training cancelled");
					return;
				}
				EndDrawing();
			}
		}
	}
}

void saveGrid(void) {
	const std::string directory = "traindata/";
	const std::string subdirectory = std::to_string(cnnState.currentLabel);
	try {
		if (!std::filesystem::exists(directory)) {
			std::filesystem::create_directory(directory);
		}
		if (!std::filesystem::exists(directory + subdirectory)) {
			std::filesystem::create_directory(directory + subdirectory);
		}
	} catch (const std::exception &error) {
		std::cerr << "ERROR: " << error.what() << std::endl;
	}
	for (int number = 0; number < 100000; number++) {
		// Pad to 5 digits to match loadTrainingData expectations.
		const std::string num = std::to_string(number);
		const std::string pad = std::string(5 - num.length(), '0');
		const std::string filename =
			directory + subdirectory + "/image_" + pad + num + ".png";
		if (!std::filesystem::exists(filename)) {
			cnnState.canvas.exportImageCanvas(filename);
			break;
		}
	}
}

void convuluctionGallery(Machine &machine, std::vector<float> input) {
	const int numFilters = machine.CNN.getNumFilters();
	const int depth = machine.CNN.getNumConvLayers();

	static int currentFilter = 0;
	static int currentDepth = 0;

	while (true) {
		cnnState.canvas.clear();
		cnnState.canvas.putConvolutionResultOnCanvas(
			machine, input, currentFilter, currentDepth);
		BeginDrawing();
		ClearBackground(RAYWHITE);
		const Layout galleryLayout = Layout::calculate();
		cnnState.canvas.position = galleryLayout.canvasPos;
		drawCanvas(cnnState.canvas, galleryLayout.cellSize);
		DrawText(
			TextFormat("Filter: %d | Depth: %d", currentFilter, currentDepth),
			10, 10, 20, BLACK);
		EndDrawing();

		if (IsKeyPressed(KEY_RIGHT)) {
			currentFilter = (currentFilter + 1) % numFilters;
		}
		if (IsKeyPressed(KEY_LEFT)) {
			currentFilter = (currentFilter - 1 + numFilters) % numFilters;
		}
		if (IsKeyPressed(KEY_UP)) {
			currentDepth = (currentDepth + 1) % depth;
		}
		if (IsKeyPressed(KEY_DOWN)) {
			currentDepth = (currentDepth - 1 + depth) % depth;
		}
		if (IsKeyPressed(KEY_ESCAPE)) {
			break;
		}
	}
	cnnState.canvas.clear();
}

static int epoch = 10;

static void handleKeyboardInput(Machine &machine) {
	if (!IsWindowFocused()) {
		return;
	}

	// Clear canvas
	if (IsKeyPressed(KEY_C)) {
		cnnState.canvas.clear();
		cnnState.isPredicting = false;
		TraceLog(LOG_INFO, "Canvas cleared");
	}

	// Train current digit — centre first so the model sees MNIST-aligned input.
	if (IsKeyPressed(KEY_T)) {
		cnnState.canvas.centreDigit();
		std::vector<float> input = cnnState.canvas.toVector();
		std::vector<float> target(10, 0.0f);
		target[cnnState.currentLabel] = 1.0f;

		// Train multiple times for better learning
		for (int i = 0; i < 10; ++i) {
			machine.CNN.train(input, target);
		}
		TraceLog(LOG_INFO, "Trained digit: %d", cnnState.currentLabel);
	}

	if (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyDown(KEY_LEFT_SHIFT) &&
		IsKeyPressed(KEY_S)) {
		TraceLog(LOG_INFO, "Saved grid for training.");
		saveGrid();
	}

	if (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_T)) {
		trainModel(machine, epoch);
		TraceLog(LOG_INFO, "Trained digit: %d", epoch);
	}

	// Conv on Grid Gallery
	if (IsKeyPressed(KEY_H)) {
		std::vector<float> input = cnnState.canvas.toVector();
		convuluctionGallery(machine, input);
		TraceLog(LOG_INFO, "Applied convolution on canvas");
	}

	// Predict — centre digit first so inference matches MNIST layout,
	// then run the network.
	if (IsKeyPressed(KEY_P)) {
		cnnState.canvas.centreDigit();
		std::vector<float> input = cnnState.canvas.toVector();
		cnnState.predictions = machine.CNN.feedForward(input);
		cnnState.isPredicting = true;

		// Find best prediction
		int	  maxIdx = 0;
		float maxVal = cnnState.predictions[0];
		for (int i = 1; i < 10; ++i) {
			if (cnnState.predictions[i] > maxVal) {
				maxVal = cnnState.predictions[i];
				maxIdx = i;
			}
		}
		TraceLog(LOG_INFO, "Predicted: %d (%.1f%% confidence)", maxIdx,
				 maxVal * 100);
	}

	// Reset network
	if (IsKeyPressed(KEY_R)) {
		const float learnRate = machine.CNN.getClassifier().getLearnRate();
		machine.CNN = ConvNeuralNetwork(
			machine.CNN.getInputWidth(), machine.CNN.getInputHeight(),
			machine.CNN.getNumFilters(), machine.CNN.getNumConvLayers(),
			machine.CNN.getKernelSize(),
			machine.CNN.getClassifier().getNumberOfHiddenNodes(),
			machine.CNN.getClassifier().getNumberOfOutputsNodes(),
			machine.CNN.getClassifier().getHiddenLayerLength());
		machine.CNN.getClassifier().setLearnRate(learnRate);
		cnnState.isPredicting = false;
		TraceLog(LOG_INFO, "CNN network reset");
	}

	// Change current label (0-9)
	for (int i = KEY_ZERO; i <= KEY_NINE; ++i) {
		if (IsKeyPressed(i)) {
			cnnState.currentLabel = i - KEY_ZERO;
			TraceLog(LOG_INFO, "Current label: %d", cnnState.currentLabel);
		}
	}

	// Save/Load model
	if (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_S)) {
		try {
			ConvNeuralNetwork::serialize(machine.CNN, "ConvNeuralNetwork.json");
			TraceLog(LOG_INFO, "Model saved to CNN.json");
		} catch (const std::exception &error) {
			TraceLog(LOG_ERROR, "Save failed: %s", error.what());
		}
	}

	if (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_L)) {
		try {
			machine.CNN =
				ConvNeuralNetwork::deserialize("ConvNeuralNetwork.json");
			cnnState.isPredicting = false;
			TraceLog(LOG_INFO, "Model loaded from CNN.json");
		} catch (const std::runtime_error &error) {
			TraceLog(LOG_WARNING, "Load failed: %s", error.what());
		}
	}
}

int handleCNNState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) {
		return STATE::MENU::MAIN;
	}

	static Button increaseEpochsButton(
		GetScreenWidth() - 220, GetScreenHeight() - 50, 100, 30, "Epochs +");
	static Button decreaseEpochsButton(
		GetScreenWidth() - 110, GetScreenHeight() - 50, 100, 30, "Epochs -");
	static bool configuredButtons = false;
	if (!configuredButtons) {
		increaseEpochsButton.setOnClick([]() {
			epoch += 1;
			TraceLog(LOG_INFO, "Epochs set to: %d", epoch);
		});
		decreaseEpochsButton.setOnClick([]() {
			epoch = std::max(1, epoch - 1);
			TraceLog(LOG_INFO, "Epochs set to: %d", epoch);
		});
		configuredButtons = true;
	}

	increaseEpochsButton.update();
	decreaseEpochsButton.update();

	// Compute layout ONCE per frame — this is the single source of truth for
	// cell size and canvas position. Nothing else may call
	// GetScreenWidth/Height or getCellSize() independently this frame.
	const Layout layout = Layout::calculate();

	// Always keep canvas position in sync with the current layout.
	// This handles window resizes automatically without any flag machinery.
	// The position is updated BEFORE input handling so the mouse coordinate
	// conversion in handleCanvasInput uses the exact same position that will
	// be used for rendering — the canvas never moves under the cursor.
	cnnState.canvas.position = layout.canvasPos;

	// Handle input — pass layout.cellSize so input and render use identical
	// values.
	handleCanvasInput(cnnState.canvas, layout.cellSize);
	handleKeyboardInput(machine);

	// Render
	BeginDrawing();
	ClearBackground(RAYWHITE);

	// Title
	constexpr char TITLE[] = "Digit Recognition Training";
	const int	   titleSize = std::max(20, GetScreenWidth() / 40);
	const int	   titleWidth = MeasureText(TITLE, titleSize);
	DrawText(TITLE, (GetScreenWidth() - titleWidth) / 2, layout.titleY,
			 titleSize, BLACK);

	// Draw canvas
	drawCanvas(cnnState.canvas, layout.cellSize);

	// Draw predictions
	if (cnnState.isPredicting) {
		drawPredictions(cnnState.predictions, layout.predictionPos,
						!layout.showPredictionsSide);
	}

	// Instructions
	const int	   instrSize = std::max(14, GetScreenWidth() / 60);
	const Vector2 &instrPos = layout.instructionPos;
	const float	   lineSpacing = instrSize + 8;

	DrawText(TextFormat("Current Label: %d", cnnState.currentLabel), instrPos.x,
			 instrPos.y, instrSize + 2, BLACK);
	DrawText("Keys: 0-9 (Label) | C (Clear) | T (Train+Centre) | P "
			 "(Predict+Centre) | R (Reset) | H (Conv Gallery)",
			 instrPos.x, instrPos.y + lineSpacing, instrSize, DARKGRAY);
	DrawText("Mouse: Left (Draw soft) | Right (Erase) | Ctrl+Shift+S (Save to "
			 "Database)",
			 instrPos.x, instrPos.y + lineSpacing * 2, instrSize, DARKGRAY);
	const std::string trainInfo = "Ctrl+S (Save AI) | Ctrl+L (Load AI) | "
								  "Ctrl+T (Train on Database) epoch: " +
								  std::to_string(epoch);
	DrawText(trainInfo.c_str(), instrPos.x, instrPos.y + lineSpacing * 3,
			 instrSize, DARKGRAY);

	// Back button
	backButton(machine).draw();
	increaseEpochsButton.draw();
	decreaseEpochsButton.draw();

	DrawFPS(drawFpsPos.x, drawFpsPos.y);

	EndDrawing();
	return (machine.state);
}
