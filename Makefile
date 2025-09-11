#
# Gerson Carvalho
# Machine Learnig v0.1b
#

CXX = g++
CXXFLAGS = -Wall -Wextra -Werror -g
STDRULE = -std=c++17
NAME = machinelearn
DEBUG = valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --suppressions=./valgrind.supp -s

IMGUISRC =	./includes/imgui/imgui_demo.cpp		\
			./includes/imgui/imgui_tables.cpp	\
			./includes/imgui/rlImGui.cpp		\
			./includes/imgui/imgui.cpp			\
			./includes/imgui/imgui_draw.cpp		\
			./includes/imgui/imgui_widgets.cpp

IMGUIINC =	./includes/imgui/imgui.h					\
			./includes/imgui/imconfig.h					\
			./includes/imgui/imgui_impl_raylib.h		\
			./includes/imgui/imgui_internal.h			\
			./includes/imgui/imstb_rectpack.h			\
			./includes/imgui/imstb_textedit.h			\
			./includes/imgui/imstb_truetype.h			\
			./includes/imgui/rlgl.h						\
			./includes/imgui/rlImGui.h					\
			./includes/imgui/rlImGuiColors.h			\
			./includes/imgui/extras/IconsFontAwesome6.h	\
			./includes/imgui/extras/FA6FreeSolidFontData.h

SRC =	./classes/DMatrix.cpp					\
		./classes/Perceptron.cpp				\
		./classes/NeuralNetwork.cpp				\
		./srcs/states/mainMenu.cpp				\
		./srcs/states/perceptronState.cpp		\
		./srcs/states/neuralNetworkState.cpp	\
		./srcs/input.cpp						\
		./srcs/menu.cpp							\
		./srcs/main.cpp

INCS =	-I./classes/							\
		-L./classes/							\
		-I./includes							\
		-L./includes/							\
		-I./includes/imgui/						\
		-L./includes/imgui/						\
		-I./includes/imgui/extras/				\
		-L./includes/imgui/extras/				\
		./includes/raylib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11

INCLUDES = ./includes/Machine.hpp		\
		   ./classes/NeuralNetwork.hpp	\
		   ./classes/Perceptron.hpp		\
		   ./classes/JsonParser.hpp		\
		   ./classes/DMatrix.hpp

IMGUI = $(IMGUISRC:.cpp=.o)
OBJS = $(SRC:.cpp=.o)
RM = rm -f

all: $(NAME)

$(NAME): $(IMGUI) $(OBJS)
	$(CXX) $(CXXFLAGS) $(STDRULE) $(IMGUI) $(OBJS) -o $(NAME) $(INCS)

$(OBJS): $(INCLUDES)

$(IMGUI): $(IMGUIINC)

clean:
	$(RM) $(IMGUI) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all

run: all
	./$(NAME)

debug:
	$(DEBUG) ./$(NAME)

.PHONY: all clean fclean re run debug
