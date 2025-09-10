#include "../includes/Machine.hpp"

static void engineSettings(Machine &machine) {
	ImGui::Text("Line Settings");
	ImGui::Separator();
	ImGui::InputFloat("Inclination", &machine.line.m);
	ImGui::InputFloat("Offset", &machine.line.d);
	for (size_t i = 0; i < machine.points.size(); i++) {
		const float lineY =
			calcDeclive(machine.line.m, machine.points[i].x, machine.line.d);
		machine.desired[i] = machine.points[i].y > lineY ? 1 : -1;
	}
}

void renderImGui(Machine &machine) {
	rlImGuiBegin();
	ImGui::Begin("Engine Settings");
	engineSettings(machine);
	ImGui::End();
	rlImGuiEnd();
}
