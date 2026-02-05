#ifndef BUTTON_H
#define BUTTON_H

#include "../../includes/raylib/raylib.h"
#include "../../includes/raylib/raymath.h"
#include <functional>
#include <string>

class Button {
  public:
	// Constructor
	Button(float x, float y, float width, float height,
		   const std::string &text);

	// Update and draw methods
	void Update();
	void Draw();

	// Setters for customization
	void SetText(const std::string &text);
	void SetPosition(float x, float y);
	void SetSize(float width, float height);
	void SetColors(Color normal, Color hover, Color pressed);
	void SetTextColor(Color color);
	void SetFontSize(int size);
	void SetBorderWidth(float width);
	void SetBorderColor(Color color);
	void SetRoundness(float roundness);
	void SetOnClick(std::function<void()> callback);

	// Getters
	bool	  IsPressed() const;
	bool	  IsHovered() const;
	Rectangle GetBounds() const;

	// State management
	void SetEnabled(bool enabled);
	bool IsEnabled() const;

  private:
	// Position and size
	Rectangle bounds;

	// Text properties
	std::string text;
	int			fontSize;
	Color		textColor;

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
	Color GetCurrentColor() const;
	void  CheckInteraction();
};

#endif // BUTTON_H
