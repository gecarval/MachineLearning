#include "../includes/Machine.hpp"

// Window Settings
static const char windowTitle[] = "Machine Learning";
static const unsigned int windowWidth = 1200;
static const unsigned int windowHeight = 800;
static const unsigned int frameLimit = 120;
static const Vector2 drawFpsPos = (Vector2) {10.0f, 10.0f};

// Machine Camera2D Settings
static const float posX = windowWidth / 2.0f;
static const float posY = windowHeight / 2.0f;
static const Vector2 screenMiddle = (Vector2) {posX, posY};
static const Vector2 target = screenMiddle;
static const Vector2 offset = screenMiddle;
static const float rotation = 0.0f;
static const float zoom = 1.0f;

// Render Texture Settings
static const Color backGroundColor = RAYWHITE;

// Simulation Settings
static const Line initialLine = {500.0f, 0.5f, -200.0f};
static const unsigned int initialPointAmount = 5000;
static const unsigned int pointRadius = 3;
static const Color pointColor = BLACK;

float calcDeclive(float m, float x, float d)
{
	return (m * x + d);
}

void initParticles(Machine &machine)
{
	machine.points = std::vector<Vector2>(initialPointAmount);
	machine.desired = std::vector<float>(initialPointAmount);
	for (size_t i = 0; i < initialPointAmount; i++)
	{
		const float randXPos = GetRandomValue(-windowWidth, windowWidth);
		const float randYPos = GetRandomValue(-windowHeight, windowHeight);
		machine.points[i] = (Vector2) {randXPos, randYPos};
		const float lineY = calcDeclive(initialLine.m, randXPos, initialLine.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

void initEngine(Machine &machine)
{
	machine.line = initialLine;
	machine.camera = (Camera2D) {offset, target, rotation, zoom};
	InitWindow(windowWidth, windowHeight, windowTitle);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	initParticles(machine);
}

void DrawAxis(Machine &machine)
{
	static const float thick = 2;
	static const float x = 2000.0f;
	const float m = machine.line.m;
	const float d = machine.line.d;
	DrawLineEx({0, -x}, {0, x}, thick, GREEN);
	DrawLineEx({-x, 0}, {x, 0}, thick, RED);
	const float yi = calcDeclive(m, -x, d);
	const float yf = calcDeclive(m, x, d);
	DrawLineEx({-x, yi}, {x, yf}, thick, BLUE);
	const size_t index = 0;
	const float w0 = machine.brain.getWeightedX0(1, index);
	const float w1 = machine.brain.getWeightedX1(1, index);
	const float wb = machine.brain.getBias() * machine.brain.getBiasWeight();
	const float myi = (-(w0 * -x) - wb) / w1;
	const float myf = (-(w0 * x) - wb) / w1;
	DrawLineEx({-x, myi}, {x, myf}, thick, ORANGE);
}

void renderMachine(Machine &machine)
{
	BeginMode2D(machine.camera);
	ClearBackground(backGroundColor);
	for (size_t i = 0; i < initialPointAmount; i++)
	{
		const int desired = machine.desired[i];
		const Vector2 center = machine.points[i];
		const int guess = machine.brain.feedFoward(&center, 1);
		if (desired == 1)
			DrawCircleLinesV(center, pointRadius + 1, pointColor);
		else
			DrawCircleV(center, pointRadius, pointColor);
		if (guess == desired)
			DrawCircleV(center, pointRadius - 1, GREEN);
		else
			DrawCircleV(center, pointRadius - 1, RED);
	}
	DrawAxis(machine);
	EndMode2D();
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
}

void updateEngine(Machine &machine)
{
	engineInput(machine);
	BeginDrawing();
	renderMachine(machine);
	renderImGui(machine);
	EndDrawing();
}

void endEngine(Machine &machine)
{
	machine.points.clear();
	rlImGuiShutdown();
	CloseWindow();
}

int main(void)
{
	std::time_t now = std::time(0);
	std::tm *local_time = std::localtime(&now);
	SetRandomSeed(local_time->tm_sec);
	Machine machine;
	initEngine(machine);
	while (!WindowShouldClose())
		updateEngine(machine);
	endEngine(machine);
	return (0);
}
