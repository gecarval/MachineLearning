#include "../../includes/Machine.hpp"

// Main Menu Title Setting
static const char title[] = "MAIN MENU";
static const char subTitle[] = "PRESS ENTER or SPACE to START";

void renderMainMenu(Machine &machine) {
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
	machine.state = STATE::MAINMENU;
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
