#ifndef BUTTON_H
#define BUTTON_H

#include "../../includes/raylib/raylib.h"
#include "../../includes/raylib/raymath.h"
#include <functional>
#include <string>

class Button {
  private:
	// Position and size
	Rectangle bounds;

	// Text properties
	std::string	 text;
	unsigned int fontSize;
	Color		 textColor;

	// Visual properties
	Color normalColor;
	Color hoverColor;
	Color pressedColor;
	Color borderColor;
	float borderWidth;
	float roundness;

	// State
	bool isHovered;
	bool isPressed;
	bool wasPressed;
	bool enabled;

	// Callback
	std::function<void()> onClick;

	// Helper methods
	Color getCurrentColor() const;
	void  checkInteraction();

  public:
	// Constructor
	Button(const float x, const float y, const float width, const float height,
		   const std::string &text);
	Button(const float x, const float y, const Vector2 &size,
		   const std::string &text);
	Button(const Vector2 &position, const float width, const float height,
		   const std::string &text);
	Button(const Vector2 &position, const Vector2 &size,
		   const std::string &text);
	Button(const Rectangle &button, const std::string &text);

	// Update and draw methods
	void update();
	void draw() const;

	// Setters for customization
	void setText(const std::string &text);
	void setPosition(const float x, const float y);
	void setPosition(const Vector2 &position);
	void setSize(float width, float height);
	void setSize(const Vector2 &size);
	void setBounds(const Rectangle &size);
	void setColors(const Color &normal, const Color &hover,
				   const Color &pressed);
	void setTextColor(const Color &color);
	void setFontSize(const unsigned int size);
	void setBorderWidth(const float width);
	void setBorderColor(const Color &color);
	void setRoundness(const float roundness);
	void setOnClick(std::function<void()> callback);

	// Getters
	bool			 isButtonPressed() const;
	bool			 isButtonHovered() const;
	Rectangle		&getBounds();
	const Rectangle &getBounds() const;

	// State management
	void setEnabled(const bool enabled);
	bool isEnabled() const;
};

#endif // BUTTON_H
