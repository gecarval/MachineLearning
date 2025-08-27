#include "../includes/Machine.hpp"

static void engineSettings(Machine &machine) {
	static const ImVec4 TextColor = (ImVec4){0, 128, 128, 255};
	ImGui::Text("Machine Settings");
	ImGui::Separator();
	ImGui::TextColored(TextColor, "Machine Status");
	ImGui::Separator();
	ImGui::Text("Funny Button");
	if (ImGui::Button("click")) std::cout << "LMAO!" << std::endl;
	(void)machine;
}

void renderImGui(Machine &machine) {
	rlImGuiBegin();
	ImGui::Begin("Engine Settings");
	engineSettings(machine);
	ImGui::End();
	rlImGuiEnd();
}
