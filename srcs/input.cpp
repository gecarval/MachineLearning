#include "../includes/Machine.hpp"
#include <cstddef>

void engineInput(Machine &machine)
{
	const float walkSpeed = 20.0f / machine.camera.zoom;
	const float zoomDelta = GetMouseWheelMove() * machine.camera.zoom * 0.1f;
	const Vector2 mousePan = GetMouseDelta() / machine.camera.zoom;
	static const float minZoom = 0.1f;
	static const float maxZoom = 3.0f;

	if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON))
	{
		machine.camera.target -= mousePan;
	}
	if (IsKeyDown(KEY_W))
	{
		machine.camera.target.y -= walkSpeed;
	}
	if (IsKeyDown(KEY_S))
	{
		machine.camera.target.y += walkSpeed;
	}
	if (IsKeyDown(KEY_A))
	{
		machine.camera.target.x -= walkSpeed;
	}
	if (IsKeyDown(KEY_D))
	{
		machine.camera.target.x += walkSpeed;
	}
	if (IsKeyDown(KEY_T))
	{
		const size_t size = machine.points.size();
		for (size_t i = 0; i < size; i++)
		{
			const Vector2 &inputArray = machine.points[i];
			const int desired = machine.desired[i];
			machine.brain.train(inputArray, i, desired);
		}
	}
	machine.camera.zoom += zoomDelta;
	machine.camera.zoom = Clamp(machine.camera.zoom, minZoom, maxZoom);
}
