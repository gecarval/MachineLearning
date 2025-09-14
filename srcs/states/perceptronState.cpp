#include "../../includes/Machine.hpp"

// Points Draw Settings
static const unsigned int pointRadius = 5;
static const Color		  pointColor = BLACK;

static void inputHandler(Machine &machine) {
	const float	  walkSpeed = 20.0f / machine.camera.zoom;
	const float	  zoomDelta = GetMouseWheelMove() * machine.camera.zoom * 0.1f;
	const Vector2 mousePan = GetMouseDelta() / machine.camera.zoom;
	static const float minZoom = 0.1f;
	static const float maxZoom = 3.0f;

	if (IsMouseButtonDown(MOUSE_MIDDLE_BUTTON)) {
		machine.camera.target -= mousePan;
	}
	if (IsKeyDown(KEY_W)) {
		machine.camera.target.y -= walkSpeed;
	}
	if (IsKeyDown(KEY_S)) {
		machine.camera.target.y += walkSpeed;
	}
	if (IsKeyDown(KEY_A)) {
		machine.camera.target.x -= walkSpeed;
	}
	if (IsKeyDown(KEY_D)) {
		machine.camera.target.x += walkSpeed;
	}
	if (IsKeyDown(KEY_T)) {
		const size_t size = machine.points.size();
		for (size_t i = 0; i < size; i++) {
			const Vector2 &inputArray = machine.points[i];
			const int	   desired = machine.desired[i];
			machine.brain.train(inputArray, i, desired);
		}
	}
	machine.camera.zoom += zoomDelta;
	machine.camera.zoom = Clamp(machine.camera.zoom, minZoom, maxZoom);
}

static void settingsMenu(Machine &machine) {
	ImGui::Text("Line Settings");
	ImGui::Separator();
	ImGui::InputFloat("Inclination", &machine.line.m);
	ImGui::InputFloat("Offset", &machine.line.d);
	for (size_t i = 0; i < machine.points.size(); i++) {
		const float lineY =
			calcDeclive(machine.line.m, machine.points[i].x, machine.line.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

static void renderImGui(Machine &machine) {
	rlImGuiBegin();
	ImGui::Begin("Engine Settings");
	settingsMenu(machine);
	ImGui::End();
	rlImGuiEnd();
}

void renderPerceptron(Machine &machine) {
	// Constants for layout
	static const float	 thickness = 1.5f;
	static const Vector2 mockTest = Vector2Zero();

	// Position in top-right corner
	const Vector2 perceptronOffset =
		(Vector2){GetScreenWidth() - 250.0f, 50.0f};
	const Vector2 perceptronCenter = (Vector2){10.0f, 10.0f} + perceptronOffset;
	const float	  perceptronRadius = 15.0f;

	// Increased spacing for input/output lines
	const Vector2 w0LineOffset = (Vector2){perceptronRadius * -4, 30};
	const Vector2 w1LineOffset = (Vector2){perceptronRadius * -4, -30};
	const Vector2 yLineOffset = (Vector2){perceptronRadius * 4, 0};
	const Vector2 w0LinePos = perceptronCenter + w0LineOffset;
	const Vector2 w1LinePos = perceptronCenter + w1LineOffset;
	const Vector2 yLinePos = perceptronCenter + yLineOffset;

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
	const float		   x = GetScreenWidth() + GetScreenHeight();
	const float		   m = machine.line.m;
	const float		   d = machine.line.d;
	const float		   yi = calcDeclive(m, -x, d);
	const float		   yf = calcDeclive(m, x, d);
	DrawLineEx({-x, yi}, {x, yf}, thick, BLUE);
	DrawLineEx({0, -x}, {0, x}, thick, GREEN);
	DrawLineEx({-x, 0}, {x, 0}, thick, RED);
	// Perceptron Line Prediction
	const size_t index = 0;
	const float	 w0 = machine.brain.getWeightedX0(1, index);
	const float	 w1 = machine.brain.getWeightedX1(1, index);
	const float	 wb = machine.brain.getBias() * machine.brain.getBiasWeight();
	const float	 myi = (-(w0 * -x) - wb) / w1;
	const float	 myf = (-(w0 * x) - wb) / w1;
	DrawLineEx({-x, myi}, {x, myf}, thick, ORANGE);
}

void DrawPoints(Machine &machine) {
	for (size_t i = 0; i < machine.points.size(); i++) {
		const Vector2 center = machine.points[i];
		const int	  desired = machine.desired[i];
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

void renderCartesianPlane(Machine &machine) {
	BeginMode2D(machine.camera);
	ClearBackground(backGroundColor);
	DrawPoints(machine);
	DrawAxis(machine);
	EndMode2D();
}

int handlePerceptronState(Machine &machine) {
	SetExitKey(0);
	if (IsKeyPressed(KEY_ESCAPE)) return (STATE::MAINMENU);
	inputHandler(machine);
	BeginDrawing();
	renderCartesianPlane(machine);
	renderPerceptron(machine);
	renderImGui(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (STATE::PERCEPTRON);
}
