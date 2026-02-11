#include "../../includes/Machine.hpp"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>

void renderSettings(Machine &mch) {
	// Available resolutions
	static const std::vector<Resolution> RESOLUTIONS = {
		{1920, 1080}, {1600, 900}, {1440, 900},
		{1366, 768},  {1280, 720}, {800, 600}};
	static int currentResIndex = 5; // Start at 800x600
	// Title configuration
	static const std::string TITLE = "SETTINGS";
	const int				 titleFontSize = GetScreenWidth() / 20;
	const int	  titleWidth = MeasureText(TITLE.c_str(), titleFontSize);
	const Vector2 titlePos = {(GetScreenWidth() - titleWidth) / 2.0f, 20.0f};
	// Screen center
	const Vector2 screenCenter = {GetScreenWidth() / 2.0f,
								  GetScreenHeight() / 2.0f};
	// Button configuration
	constexpr Vector2 BUTTON_SIZE = {200.0f, 50.0f};
	constexpr float	  BUTTON_SPACING = 10.0f;

	const Vector2 firstButtonPos = {screenCenter.x - BUTTON_SIZE.x / 2.0f,
									screenCenter.y - BUTTON_SIZE.y / 2.0f};
	// Lambda to apply resolution
	auto applyResolution = [](int index) {
		if (index >= 0 && index < static_cast<int>(RESOLUTIONS.size())) {
			SetWindowSize(RESOLUTIONS[index].width, RESOLUTIONS[index].height);
		}
	};
	// Create buttons
	static Button buttons[] = {
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Increase Resolution"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Decrease Resolution"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Toggle Fullscreen"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Back")};
	// Initialize callbacks once
	static bool initialized = false;
	if (!initialized) {
		buttons[0].setOnClick([&applyResolution]() {
			currentResIndex = Clamp(currentResIndex - 1, 0,
									static_cast<int>(RESOLUTIONS.size()) - 1);
			applyResolution(currentResIndex);
		});
		buttons[1].setOnClick([&applyResolution]() {
			currentResIndex = Clamp(currentResIndex + 1, 0,
									static_cast<int>(RESOLUTIONS.size()) - 1);
			applyResolution(currentResIndex);
		});
		buttons[2].setOnClick([]() { ToggleFullscreen(); });
		buttons[3].setOnClick([&mch]() { mch.state = STATE::MENU::MAIN; });
		initialized = true;
	}
	// Update button positions and state
	for (int i = 0; i < 4; ++i) {
		float yPos = firstButtonPos.y + i * (BUTTON_SIZE.y + BUTTON_SPACING);
		buttons[i].setPosition(firstButtonPos.x, yPos);
		buttons[i].update();
	}
	// Render
	ClearBackground(RAYWHITE);
	for (auto &button : buttons) {
		button.draw();
	}
	// Draw title
	DrawText(TITLE.c_str(), titlePos.x, titlePos.y, titleFontSize, DARKGREEN);
	// Draw current resolution
	const std::string currentRes = RESOLUTIONS[currentResIndex].toString();
	const int		  subTitleFontSize = GetScreenWidth() / 30;
	const int subTitleWidth = MeasureText(currentRes.c_str(), subTitleFontSize);
	const Vector2 subTitlePos = {(GetScreenWidth() - subTitleWidth) / 2.0f,
								 static_cast<float>(subTitleFontSize + 100)};
	DrawText(currentRes.c_str(), subTitlePos.x, subTitlePos.y, subTitleFontSize,
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
	// Title rendering
	static const std::string TITLE = "MAIN MENU";
	const int				 titleFontSize = GetScreenWidth() / 20;
	const int	  titleWidth = MeasureText(TITLE.c_str(), titleFontSize);
	const Vector2 titlePos = {(GetScreenWidth() - titleWidth) / 2.0f, 20.0f};
	// Button configuration
	constexpr Vector2 BUTTON_SIZE = {200.0f, 50.0f};
	constexpr float	  BUTTON_SPACING = 10.0f;
	const Vector2	  screenCenter = {GetScreenWidth() / 2.0f,
									  GetScreenHeight() / 2.0f};
	const Vector2	  firstButtonPos = {screenCenter.x - BUTTON_SIZE.x / 2.0f,
										screenCenter.y - BUTTON_SIZE.y / 2.0f};
	// Create buttons (static for persistence)
	static Button buttons[] = {
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Perceptron"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Neural Network"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Conv Neural Network"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Settings"),
		Button(firstButtonPos.x, firstButtonPos.y, BUTTON_SIZE.x, BUTTON_SIZE.y,
			   "Exit")};
	// Initialize button callbacks once
	static bool initialized = false;
	if (!initialized) {
		buttons[0].setOnClick(
			[&mch]() { mch.state = STATE::GAME::PERCEPTRON; });
		buttons[1].setOnClick(
			[&mch]() { mch.state = STATE::GAME::NEURALNETWORK; });
		buttons[2].setOnClick([&mch]() { mch.state = STATE::GAME::CNN; });
		buttons[3].setOnClick([&mch]() { mch.state = STATE::MENU::SETTING; });
		buttons[4].setOnClick([&mch]() {
			mch.points.clear();
			rlImGuiShutdown();
			CloseWindow();
			std::exit(0);
		});
		initialized = true;
	}
	// Update button positions and state
	for (int i = 0; i < 5; ++i) {
		float yPos = firstButtonPos.y + i * (BUTTON_SIZE.y + BUTTON_SPACING);
		buttons[i].setPosition(firstButtonPos.x, yPos);
		buttons[i].update();
	}
	// Render
	ClearBackground(RAYWHITE);
	for (auto &button : buttons) {
		button.draw();
	}
	DrawText(TITLE.c_str(), titlePos.x, titlePos.y, titleFontSize, DARKGREEN);
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
