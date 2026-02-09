#include "../../includes/Machine.hpp"

void renderMainMenu(Machine &machine) {
	// Main Menu Title Setting
	static const char title[] = "MAIN MENU";
	static const char subTitle[] = "PRESS ENTER or SPACE to START";
	// Main Menu Title Padding
	const int	  titleFontSize = GetScreenWidth() / 20;
	const int	  subTitleFontSize = titleFontSize / 2;
	const int	  titlePosOffset = titleFontSize * sizeof(title) / 3;
	const Vector2 screenCenter =
		(Vector2){GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
	const Vector2 titlePos = (Vector2){screenCenter.x - titlePosOffset, 20};
	const int	  subTitlePosOffset = subTitleFontSize * sizeof(subTitle) / 3;
	const Vector2 subTitlePos =
		(Vector2){screenCenter.x - subTitlePosOffset, screenCenter.y};
	ClearBackground(RAYWHITE);
	DrawText(title, titlePos.x, titlePos.y, titleFontSize, DARKGREEN);
	DrawText(subTitle, subTitlePos.x, subTitlePos.y, subTitleFontSize,
			 DARKGREEN);
}

int handleMainState(Machine &machine) {
	SetExitKey(KEY_ESCAPE);
	if (IsKeyPressed(KEY_SPACE)) return (STATE::GAME::PERCEPTRON);
	if (IsKeyPressed(KEY_ENTER)) return (STATE::GAME::NEURALNETWORK);
	BeginDrawing();
	renderMainMenu(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (machine.state);
}

int handleMenuState(Machine &machine) {
	switch (machine.state) {
		case STATE::MENU::MAIN:
			machine.state = handleMainState(machine);
			break;
		case STATE::MENU::SETTING:
			machine.state = STATE::MENU::MAIN;
			break;
	}
	return machine.state;
}
