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
	machine.state = STATE::MAINMENU;
	machine.line = initialLine;
	machine.camera = (Camera2D){offset, target, rotation, zoom};
	InitWindow(windowWidth, windowHeight, windowTitle);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	initPoints(machine);
}

void renderNeuralNetwork(Machine &machine) {
	static const char	 title[] = "Work In Progress...";
	static const char	 subTitle[] = "PRESS ESCAPE to JUMP to the MAIN MENU";
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
