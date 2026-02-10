#include "../../includes/Machine.hpp"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>

Rectangle createRectangle(const Vector2 &pos, const Vector2 &size) {
	return (Rectangle){pos.x, pos.y, size.x, size.y};
};

void renderSettings(Machine &mch) {
	// Resolution Text
	static const std::vector<std::string> resolution = {
		"1920x1080", "1600x900", "1440x900",
		"1366x768",	 "1280x720", " 800x600"};
	static int		   state = 5;
	static std::string text = resolution[state];
	// Settings Title Text
	static std::string title = "SETTINGS";
	// Settings Title Position
	const Vector2 screenCenter =
		(Vector2){GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
	const int	  titleFontSize = GetScreenWidth() / 20;
	const int	  titlePosOffset = titleFontSize * title.length() / 3;
	const Vector2 titlePos = (Vector2){screenCenter.x - titlePosOffset, 20};
	// Main Menu Button Position
	const Vector2	buttonSize = (Vector2){200, 50};
	const Vector2	buttonPos = screenCenter - buttonSize / 2;
	const int		padding2 = buttonSize.y + 10;
	const int		padding3 = 2 * buttonSize.y + 20;
	const int		padding4 = 3 * buttonSize.y + 30;
	const Rectangle bounds = createRectangle(buttonPos, buttonSize);
	static Button	button1(bounds, "Raise Resolution");
	static Button	button2(bounds, "Down Resolution");
	static Button	button3(bounds, "FullScreen");
	static Button	button4(bounds, "Back");
	button1.setPosition(buttonPos.x, buttonPos.y);
	button2.setPosition(buttonPos.x, buttonPos.y + padding2);
	button3.setPosition(buttonPos.x, buttonPos.y + padding3);
	button4.setPosition(buttonPos.x, buttonPos.y + padding4);
	button1.setOnClick([]() {
		state = Clamp(state - 1, 0, resolution.size());
		text = resolution[state];
		const int width = std::stoi(text);
		const int height = std::stoi(text.substr(5));
		SetWindowSize(width, height);
	});
	button2.setOnClick([]() {
		state = Clamp(state + 1, 0, resolution.size() - 1);
		text = resolution[state];
		const int width = std::stoi(text);
		const int height = std::stoi(text.substr(5));
		SetWindowSize(width, height);
	});
	button3.setOnClick([]() { ToggleFullscreen(); });
	button4.setOnClick([&mch]() { mch.state = STATE::MENU::MAIN; });
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
	// Sub Title Fonts
	const int	  subTitleFontSize = GetScreenWidth() / 30;
	const int	  subTitlePosOffsetX = subTitleFontSize * text.length() / 3;
	const float	  subTitlePosOffsetY = subTitleFontSize + 100;
	const Vector2 subTitlePos =
		(Vector2){screenCenter.x - subTitlePosOffsetX, subTitlePosOffsetY};
	DrawText(text.c_str(), subTitlePos.x, subTitlePos.y, subTitleFontSize,
			 DARKGREEN);
}

int handleSettingState(Machine &machine) {
	SetExitKey(KEY_ESCAPE);
	BeginDrawing();
	renderSettings(machine);
	DrawFPS(drawFpsPos.x, drawFpsPos.y);
	EndDrawing();
	return (machine.state);
}

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
	const Vector2	buttonSize = (Vector2){200, 50};
	const Vector2	buttonPos = screenCenter - buttonSize / 2;
	const int		padding2 = buttonSize.y + 10;
	const int		padding3 = 2 * buttonSize.y + 20;
	const int		padding4 = 3 * buttonSize.y + 30;
	const Rectangle bounds = createRectangle(buttonPos, buttonSize);
	static Button	button1(bounds, "Perceptron");
	static Button	button2(bounds, "Neural Network");
	static Button	button3(bounds, "Settings");
	static Button	button4(bounds, "Exit");
	button1.setPosition(buttonPos.x, buttonPos.y);
	button2.setPosition(buttonPos.x, buttonPos.y + padding2);
	button3.setPosition(buttonPos.x, buttonPos.y + padding3);
	button4.setPosition(buttonPos.x, buttonPos.y + padding4);
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
			machine.state = handleSettingState(machine);
			break;
	}
	return machine.state;
}
