#include "../includes/Machine.hpp"
#include <cstdlib>

// Window Settings
static const char		windowTitle[] = "Machine Learning";
static const int		windowWidth = 1200;
static const int		windowHeight = 800;
static const int		frameLimit = 120;
static const Vector2	drawFpsPos = (Vector2){10.0f, 10.0f};

// Machine Camera2D Settings
static const float	 posX = windowWidth / 2.0f;
static const float	 posY = windowHeight / 2.0f;
static const Vector2 screenMiddle = (Vector2){posX, posY};
static const Vector2 target = screenMiddle;
static const Vector2 offset = screenMiddle;
static const float	 rotation = 0.0f;
static const float	 zoom = 1.0f;

// Render Texture Settings
static const Color	 backGroundColor = RAYWHITE;

// Simulation Settings
static const int	initialPointAmount = 5000;
static const int	pointRadius = 3;
static const Color	pointColor = BLACK;

void initParticles(Machine &machine) {
	
	machine.points = std::vector<Vector2>(initialPointAmount);
	for (std::vector<Vector2>::iterator p = machine.points.begin();
		 p != machine.points.cend(); p++) {
		const float randXPos = GetRandomValue(0, windowWidth);
		const float randYPos = GetRandomValue(0, windowHeight);
		(*p) = (Vector2){randXPos, randYPos};
	}
}

void initEngine(Machine &machine) {
	InitWindow(windowWidth, windowHeight, windowTitle);
	SetTargetFPS(frameLimit);
	rlImGuiSetup(true);
	initParticles(machine);
	machine.camera = (Camera2D){offset, target, rotation, zoom};
}

void renderMachine(Machine &machine) {
	BeginMode2D(machine.camera);
	ClearBackground(backGroundColor);
	for (std::vector<Vector2>::iterator p = machine.points.begin();
		 p != machine.points.cend(); p++) {
		const Vector2 center = (*p);
		DrawCircleV(center, pointRadius, pointColor);
	}
	EndMode2D();
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
}

void updateEngine(Machine &machine) {
	engineInput(machine);
	BeginDrawing();
	renderMachine(machine);
	renderImGui(machine);
	EndDrawing();
}

void endEngine(Machine &machine) {
	machine.points.clear();
	rlImGuiShutdown();
	CloseWindow();
}

int main(void) {
	Machine machine;
	initEngine(machine);
	while (!WindowShouldClose()) updateEngine(machine);
	endEngine(machine);
	return (0);
}
