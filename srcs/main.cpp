#include "../includes/Machine.hpp"

// Window Settings
static const char windowTitle[] = "Machine Learning";
static const int windowWidth = 1200;
static const int windowHeight = 800;
static const int frameLimit = 120;
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
static const int initialPointAmount = 5000;
static const int pointRadius = 3;
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
		const float lineY = calcDeclive(0.5f, machine.points[i].x, -500);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

void initEngine(Machine &machine)
{
	InitWindow(windowWidth, windowHeight, windowTitle);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	initParticles(machine);
	machine.camera = (Camera2D) {offset, target, rotation, zoom};
}

void DrawAxis(void)
{
	static const float x = 2000;
	static const float yi = calcDeclive(0.5f, -x, -500);
	static const float yf = calcDeclive(0.5f, x, -500);
	DrawLine(-x, yi, x, yf, BLUE);
	DrawLine(0, -x, 0, x, GREEN);
	DrawLine(-x, 0, x, 0, RED);
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
	DrawAxis();
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
