#include "./Button.hpp"

// Constructor
Button::Button(const float x, const float y, const float width,
			   const float height, const std::string &text)
	: bounds{x, y, width, height}, text(text), fontSize(20), textColor(WHITE),
	  normalColor{100, 100, 100, 255}, hoverColor{130, 130, 130, 255},
	  pressedColor{70, 70, 70, 255}, borderColor{50, 50, 50, 255},
	  borderWidth(0.0f), roundness(0.0f), isHovered(false), isPressed(false),
	  wasPressed(false), enabled(true), onClick(nullptr) {
}

Button::Button(const float x, const float y, const Vector2 &size,
			   const std::string &text)
	: bounds{x, y, size.x, size.y}, text(text), fontSize(20), textColor(WHITE),
	  normalColor{100, 100, 100, 255}, hoverColor{130, 130, 130, 255},
	  pressedColor{70, 70, 70, 255}, borderColor{50, 50, 50, 255},
	  borderWidth(0.0f), roundness(0.0f), isHovered(false), isPressed(false),
	  wasPressed(false), enabled(true), onClick(nullptr) {
}

Button::Button(const Vector2 &position, const float width, const float height,
			   const std::string &text)
	: bounds{position.x, position.y, width, height}, text(text), fontSize(20),
	  textColor(WHITE), normalColor{100, 100, 100, 255},
	  hoverColor{130, 130, 130, 255}, pressedColor{70, 70, 70, 255},
	  borderColor{50, 50, 50, 255}, borderWidth(0.0f), roundness(0.0f),
	  isHovered(false), isPressed(false), wasPressed(false), enabled(true),
	  onClick(nullptr) {
}

Button::Button(const Vector2 &position, const Vector2 &size,
			   const std::string &text)
	: bounds{position.x, position.y, size.x, size.y}, text(text), fontSize(20),
	  textColor(WHITE), normalColor{100, 100, 100, 255},
	  hoverColor{130, 130, 130, 255}, pressedColor{70, 70, 70, 255},
	  borderColor{50, 50, 50, 255}, borderWidth(0.0f), roundness(0.0f),
	  isHovered(false), isPressed(false), wasPressed(false), enabled(true),
	  onClick(nullptr) {
}

Button::Button(const Rectangle &bounds, const std::string &text)
	: bounds{bounds}, text(text), fontSize(20), textColor(WHITE),
	  normalColor{100, 100, 100, 255}, hoverColor{130, 130, 130, 255},
	  pressedColor{70, 70, 70, 255}, borderColor{50, 50, 50, 255},
	  borderWidth(0.0f), roundness(0.0f), isHovered(false), isPressed(false),
	  wasPressed(false), enabled(true), onClick(nullptr) {
}

#include <iostream>

// Update method - checks for mouse interaction
void Button::update() {
	if (!this->enabled) {
		this->isHovered = false;
		this->isPressed = false;
		return;
	}

	this->checkInteraction();

	// Detect click event (button was pressed and now released)
	if (this->wasPressed && !this->isPressed && this->isHovered &&
		this->onClick) {
		this->onClick();
	}

	this->wasPressed = this->isPressed;
}

// Draw method - renders the button
void Button::draw() const {
	const Color &currentColor = this->getCurrentColor();

	// Draw button background
	if (this->roundness > 0.0f) {
		DrawRectangleRounded(this->bounds, this->roundness, 16, currentColor);
	} else {
		DrawRectangleRec(this->bounds, currentColor);
	}

	// Draw border if enabled
	if (this->borderWidth > 0.0f) {
		if (this->roundness > 0.0f) {
			/*DrawRectangleRoundedLines(this->bounds, this->roundness, 16,
									  this->borderWidth, this->borderColor);*/
			DrawRectangleRounded(this->bounds, this->roundness, 16,
								 this->borderColor);
		} else {
			DrawRectangleLinesEx(this->bounds, this->borderWidth,
								 this->borderColor);
		}
	}

	// Draw text centered
	const Vector2 textSize =
		MeasureTextEx(GetFontDefault(), this->text.c_str(), this->fontSize, 1);
	const Vector2 textPos = {
		this->bounds.x + (this->bounds.width - textSize.x) / 2.0f,
		this->bounds.y + (this->bounds.height - textSize.y) / 2.0f};

	DrawTextEx(GetFontDefault(), this->text.c_str(), textPos, this->fontSize, 1,
			   this->textColor);

	// Optional: Draw disabled overlay
	if (!this->enabled) {
		const Color &overlay = {0, 0, 0, 100};
		if (this->roundness > 0.0f) {
			DrawRectangleRounded(this->bounds, this->roundness, 16, overlay);
		} else {
			DrawRectangleRec(this->bounds, overlay);
		}
	}
}

// Setters
void Button::setText(const std::string &text) {
	this->text = text;
}

void Button::setPosition(const float x, const float y) {
	this->bounds.x = x;
	this->bounds.y = y;
}

void Button::setPosition(const Vector2 &position) {
	this->bounds.x = position.x;
	this->bounds.y = position.y;
}

void Button::setSize(const float width, const float height) {
	this->bounds.width = width;
	this->bounds.height = height;
}

void Button::setSize(const Vector2 &size) {
	this->bounds.width = size.x;
	this->bounds.height = size.y;
}

void Button::setBounds(const Rectangle &bounds) {
	this->bounds = bounds;
}

void Button::setColors(const Color &normal, const Color &hover,
					   const Color &pressed) {
	this->normalColor = normal;
	this->hoverColor = hover;
	this->pressedColor = pressed;
}

void Button::setTextColor(const Color &color) {
	this->textColor = color;
}

void Button::setFontSize(const unsigned int size) {
	this->fontSize = size;
}

void Button::setBorderWidth(const float width) {
	this->borderWidth = width;
}

void Button::setBorderColor(const Color &color) {
	this->borderColor = color;
}

void Button::setRoundness(const float roundness) {
	this->roundness = Clamp(roundness, 0.0f, 1.0f);
}

void Button::setOnClick(std::function<void()> callback) {
	this->onClick = callback;
}

// Getters
bool Button::isButtonPressed() const {
	return this->isPressed;
}

bool Button::isButtonHovered() const {
	return this->isHovered;
}

Rectangle &Button::getBounds() {
	return this->bounds;
}

const Rectangle &Button::getBounds() const {
	return this->bounds;
}

void Button::setEnabled(const bool enabled) {
	this->enabled = enabled;
}

bool Button::isEnabled() const {
	return this->enabled;
}

// Private helper methods
Color Button::getCurrentColor() const {
	if (!this->enabled) {
		return this->normalColor;
	}

	if (this->isPressed) {
		return this->pressedColor;
	} else if (this->isHovered) {
		return this->hoverColor;
	}
	return this->normalColor;
}

void Button::checkInteraction() {
	const Vector2 &mousePos = GetMousePosition();
	this->isHovered = CheckCollisionPointRec(mousePos, this->bounds);

	if (this->isHovered && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
		this->isPressed = true;
	} else if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
		this->isPressed = false;
	}
}
