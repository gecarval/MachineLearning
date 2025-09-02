#include "../includes/Machine.hpp"

static void engineSettings(Machine &machine) {
	static const ImVec4 TextColor = (ImVec4){0, 128, 128, 255};
	float biasW = machine.brain.getBiasWeight();
	float w0 = machine.brain.getWeightedX0(1, 0);
	float w1 = machine.brain.getWeightedX1(1, 0);
	ImGui::Text("Machine Settings");
	ImGui::Separator();
	ImGui::TextColored(TextColor, "Machine Status");
	ImGui::InputFloat("BiasWeight", &biasW);
	ImGui::InputFloat("W0", &w0);
	ImGui::InputFloat("W1", &w1);
	ImGui::Separator();
	ImGui::Text("Machine Settings");
	ImGui::Separator();
	ImGui::SliderFloat("Offset", &machine.line.d, -1000, 1000);
	ImGui::SliderFloat("Declive", &machine.line.m, -10.0f, 10.0f);
	ImGui::Text("Funny Button");
	if (ImGui::Button("click")) std::cout << "LMAO!" << std::endl;
	for (size_t i = 0; i < machine.points.size(); i++)
	{
		const float lineY = calcDeclive(machine.line.m, machine.points[i].x, machine.line.d);
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
