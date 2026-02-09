#include "../../includes/Machine.hpp"

void renderMainMenu(Machine &mch) {
	// Main Menu Title Text
	static std::string title = "MAIN MENU";
	// Main Menu Title Position
	const int	  titleFontSize = GetScreenWidth() / 20;
	const int	  titlePosOffset = titleFontSize * title.length() / 3;
	const Vector2 screenCenter =
		(Vector2){GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
	const Vector2 titlePos = (Vector2){screenCenter.x - titlePosOffset, 20};
	// Main Menu Button Position
	const Vector2 buttonSize = (Vector2){200, 50};
	const Vector2 buttonPos = screenCenter - buttonSize / 2;
	const int	  padding2 = buttonSize.y + 10;
	const int	  padding3 = 2 * buttonSize.y + 20;
	const int	  padding4 = 3 * buttonSize.y + 30;
	static Button button1(buttonPos.x, buttonPos.y, buttonSize, "Perceptron");
	static Button button2(buttonPos.x, buttonPos.y + padding2, buttonSize,
						  "Neural Network");
	static Button button3(buttonPos.x, buttonPos.y + padding3, buttonSize,
						  "Settings");
	static Button button4(buttonPos.x, buttonPos.y + padding4, buttonSize,
						  "Exit");
	button1.setOnClick([&mch]() { mch.state = STATE::GAME::PERCEPTRON; });
	button2.setOnClick([&mch]() { mch.state = STATE::GAME::NEURALNETWORK; });
	button3.setOnClick([&mch]() { mch.state = STATE::MENU::SETTING; });
	button4.setOnClick([&mch]() {
		mch.points.clear();
		rlImGuiShutdown();
		CloseWindow();
		std::exit(0);
	});
	button1.update();
	button2.update();
	button3.update();
	button4.update();
	ClearBackground(RAYWHITE);
	button1.draw();
	button2.draw();
	button3.draw();
	button4.draw();
	DrawText(title.c_str(), titlePos.x, titlePos.y, titleFontSize, DARKGREEN);
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
