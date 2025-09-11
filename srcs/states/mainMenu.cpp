#include "../../includes/Machine.hpp"

// Main Menu Title Setting
static const char	 title[] = "MAIN MENU";
static const char	 subTitle[] = "PRESS ENTER or SPACE to START";
static const int	 titleFontSize = windowHeight / 20;
static const int	 subTitleFontSize = titleFontSize / 2;
static const int	 titlePosOffset = titleFontSize * sizeof(title) / 3;
static const Vector2 titlePos = (Vector2){screenMiddle.x - titlePosOffset, 20};
static const int subTitlePosOffset = subTitleFontSize * sizeof(subTitle) / 3;
static const Vector2 subTitlePos =
	(Vector2){screenMiddle.x - subTitlePosOffset, screenMiddle.y};

void renderMainMenu(Machine &machine) {
	ClearBackground(backGroundColor);
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
