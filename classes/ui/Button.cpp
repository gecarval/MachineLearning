#include "./Button.hpp"

// Constructor
Button::Button(float x, float y, float width, float height,
			   const std::string &text)
	: bounds{x, y, width, height}, text(text), fontSize(20), textColor(WHITE),
	  normalColor{100, 100, 100, 255}, hoverColor{130, 130, 130, 255},
	  pressedColor{70, 70, 70, 255}, borderColor{50, 50, 50, 255},
	  borderWidth(0.0f), roundness(0.0f), isHovered(false), isPressed(false),
	  wasPressed(false), enabled(true), onClick(nullptr) {
}

// Update method - checks for mouse interaction
void Button::Update() {
	if (!enabled) {
		isHovered = false;
		isPressed = false;
		return;
	}

	CheckInteraction();

	// Detect click event (button was pressed and now released)
	if (wasPressed && !isPressed && isHovered && onClick) {
		onClick();
	}

	wasPressed = isPressed;
}

// Draw method - renders the button
void Button::Draw() {
	Color currentColor = GetCurrentColor();

	// Draw button background
	if (roundness > 0.0f) {
		DrawRectangleRounded(bounds, roundness, 16, currentColor);
	} else {
		DrawRectangleRec(bounds, currentColor);
	}

	// Draw border if enabled
	if (borderWidth > 0.0f) {
		if (roundness > 0.0f) {
			/*DrawRectangleRoundedLines(bounds, roundness, 16, borderWidth,
									  borderColor);*/
			DrawRectangleLinesEx(bounds, borderWidth, borderColor);
		} else {
			DrawRectangleLinesEx(bounds, borderWidth, borderColor);
		}
	}

	// Draw text centered
	Vector2 textSize =
		MeasureTextEx(GetFontDefault(), text.c_str(), fontSize, 1);
	Vector2 textPos = {bounds.x + (bounds.width - textSize.x) / 2.0f,
					   bounds.y + (bounds.height - textSize.y) / 2.0f};

	DrawTextEx(GetFontDefault(), text.c_str(), textPos, fontSize, 1, textColor);

	// Optional: Draw disabled overlay
	if (!enabled) {
		Color overlay = {0, 0, 0, 100};
		if (roundness > 0.0f) {
			DrawRectangleRounded(bounds, roundness, 16, overlay);
		} else {
			DrawRectangleRec(bounds, overlay);
		}
	}
}

// Setters
void Button::SetText(const std::string &text) {
	this->text = text;
}

void Button::SetPosition(float x, float y) {
	bounds.x = x;
	bounds.y = y;
}

void Button::SetSize(float width, float height) {
	bounds.width = width;
	bounds.height = height;
}

void Button::SetColors(Color normal, Color hover, Color pressed) {
	normalColor = normal;
	hoverColor = hover;
	pressedColor = pressed;
}

void Button::SetTextColor(Color color) {
	textColor = color;
}

void Button::SetFontSize(int size) {
	fontSize = size;
}

void Button::SetBorderWidth(float width) {
	borderWidth = width;
}

void Button::SetBorderColor(Color color) {
	borderColor = color;
}

void Button::SetRoundness(float roundness) {
	this->roundness = Clamp(roundness, 0.0f, 1.0f);
}

void Button::SetOnClick(std::function<void()> callback) {
	onClick = callback;
}

// Getters
bool Button::IsPressed() const {
	return isPressed;
}

bool Button::IsHovered() const {
	return isHovered;
}

Rectangle Button::GetBounds() const {
	return bounds;
}

void Button::SetEnabled(bool enabled) {
	this->enabled = enabled;
}

bool Button::IsEnabled() const {
	return enabled;
}

// Private helper methods
Color Button::GetCurrentColor() const {
	if (!enabled) {
		return normalColor;
	}

	if (isPressed) {
		return pressedColor;
	} else if (isHovered) {
		return hoverColor;
	}
	return normalColor;
}

void Button::CheckInteraction() {
	Vector2 mousePos = GetMousePosition();
	isHovered = CheckCollisionPointRec(mousePos, bounds);

	if (isHovered && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
		isPressed = true;
	} else if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
		isPressed = false;
	}
}
