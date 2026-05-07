#
# Gerson Carvalho
# Machine Learning v0.3b
#

# Compiler settings
CXX      := g++
CXXFLAGS := -Wall -Wextra -Werror -g -O3 -march=native -ffast-math \
            -ftree-vectorize -fopenmp -std=c++17

# Project settings
NAME      := machinelearn
BUILD_DIR := build

# Debugger
DEBUG := valgrind --leak-check=full --show-leak-kinds=all \
         --track-origins=yes --suppressions=./valgrind.supp -s

# Colors
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
NC     := \033[0m

# ------------------------------------------------------------
# Sources — listed explicitly so there are no path surprises
# ------------------------------------------------------------

IMGUI_SRCS := includes/imgui/imgui.cpp \
              includes/imgui/imgui_demo.cpp \
              includes/imgui/imgui_draw.cpp \
              includes/imgui/imgui_tables.cpp \
              includes/imgui/imgui_widgets.cpp \
              includes/imgui/rlImGui.cpp

PROJECT_SRCS := classes/DMatrix.cpp \
                classes/Perceptron.cpp \
                classes/NeuralNetwork.cpp \
                classes/CNN.cpp \
                classes/ui/Button.cpp \
                srcs/states/mainMenu.cpp \
                srcs/states/perceptronState.cpp \
                srcs/states/neuralNetworkState.cpp \
                srcs/states/CNNState.cpp \
                srcs/main.cpp

# ------------------------------------------------------------
# Headers (project headers only — changing these triggers rebuild)
# ------------------------------------------------------------

HEADERS := includes/Machine.hpp \
           includes/json.hpp \
           classes/DMatrix.hpp \
           classes/Perceptron.hpp \
           classes/NeuralNetwork.hpp \
           classes/CNN.hpp \
           classes/JsonParser.hpp \
           classes/ui/Button.hpp

# ------------------------------------------------------------
# Include flags
# ------------------------------------------------------------

INC_FLAGS := -Iclasses \
             -Iincludes \
             -Iincludes/imgui \
             -Iincludes/imgui/extras

# ------------------------------------------------------------
# Libraries
# ------------------------------------------------------------

LIBS := includes/raylib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11

# ------------------------------------------------------------
# Object files — mirror source tree under OBJ_DIR
# ------------------------------------------------------------

IMGUI_OBJS   := $(patsubst %.cpp,%.o,$(IMGUI_SRCS))
PROJECT_OBJS := $(patsubst %.cpp,%.o,$(PROJECT_SRCS))
ALL_OBJS     := $(IMGUI_OBJS) $(PROJECT_OBJS)

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

.DEFAULT_GOAL := all

all: $(NAME)
	@echo "$(GREEN)✓ Build complete: $(NAME)$(NC)"

$(NAME): $(ALL_OBJS)
	@echo "$(YELLOW)Linking $(NAME)...$(NC)"
	@$(CXX) $(CXXFLAGS) $(ALL_OBJS) -o $@ $(LIBS)

# ImGui objects — no dependency on project headers
$(OBJ_DIR)/includes/imgui/%.o: includes/imgui/%.cpp
	@mkdir -p $(dir $@)
	@echo "$(YELLOW)Compiling $<...$(NC)"
	@$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# classes/ objects
$(OBJ_DIR)/classes/%.o: classes/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "$(YELLOW)Compiling $<...$(NC)"
	@$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# srcs/ objects
$(OBJ_DIR)/srcs/%.o: srcs/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "$(YELLOW)Compiling $<...$(NC)"
	@$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

clean:
	@echo "$(RED)Cleaning object files...$(NC)"
	@rm -rf $(BUILD_DIR)

fclean: clean
	@echo "$(RED)Cleaning executable...$(NC)"
	@rm -f $(NAME)

re: fclean all

run: all
	@echo "$(GREEN)Running $(NAME)...$(NC)"
	@./$(NAME)

debug: all
	@echo "$(GREEN)Running $(NAME) with valgrind...$(NC)"
	@$(DEBUG) ./$(NAME)

verbose: CXXFLAGS += -v
verbose: re

help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  clean   - Remove object files"
	@echo "  fclean  - Remove object files and executable"
	@echo "  re      - Rebuild everything"
	@echo "  run     - Build and run the project"
	@echo "  debug   - Run with valgrind"
	@echo "  verbose - Build with verbose compiler output"
	@echo "  help    - Show this help message"

.PHONY: all clean fclean re run debug verbose help
