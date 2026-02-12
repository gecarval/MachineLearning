#
# Gerson Carvalho
# Machine Learning v0.3b
#

# Compiler settings
CXX := g++
CXXFLAGS := -Wall -Wextra -Werror -g -O3 -march=native -ffast-math \
            -ftree-vectorize -fopenmp -std=c++17

# Project settings
NAME := machinelearn
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# Debugger
DEBUG := valgrind --leak-check=full --show-leak-kinds=all \
         --track-origins=yes --suppressions=./valgrind.supp -s

# Directories
IMGUI_DIR := ./includes/imgui
IMGUI_EXTRAS := $(IMGUI_DIR)/extras
CLASS_DIR := ./classes
SRC_DIR := ./srcs
INCLUDES_DIR := ./includes

# ImGui sources
IMGUI_SRC := $(IMGUI_DIR)/imgui_demo.cpp \
             $(IMGUI_DIR)/imgui_tables.cpp \
             $(IMGUI_DIR)/rlImGui.cpp \
             $(IMGUI_DIR)/imgui.cpp \
             $(IMGUI_DIR)/imgui_draw.cpp \
             $(IMGUI_DIR)/imgui_widgets.cpp

# ImGui headers
IMGUI_INC := $(IMGUI_DIR)/imgui.h \
             $(IMGUI_DIR)/imconfig.h \
             $(IMGUI_DIR)/imgui_impl_raylib.h \
             $(IMGUI_DIR)/imgui_internal.h \
             $(IMGUI_DIR)/imstb_rectpack.h \
             $(IMGUI_DIR)/imstb_textedit.h \
             $(IMGUI_DIR)/imstb_truetype.h \
             $(IMGUI_DIR)/rlgl.h \
             $(IMGUI_DIR)/rlImGui.h \
             $(IMGUI_DIR)/rlImGuiColors.h \
             $(IMGUI_EXTRAS)/IconsFontAwesome6.h \
             $(IMGUI_EXTRAS)/FA6FreeSolidFontData.h

# Project sources
SRC := $(CLASS_DIR)/DMatrix.cpp \
       $(CLASS_DIR)/Perceptron.cpp \
       $(CLASS_DIR)/NeuralNetwork.cpp \
       $(CLASS_DIR)/CNN.cpp \
       $(CLASS_DIR)/ui/Button.cpp \
       $(SRC_DIR)/states/mainMenu.cpp \
       $(SRC_DIR)/states/perceptronState.cpp \
       $(SRC_DIR)/states/neuralNetworkState.cpp \
       $(SRC_DIR)/states/CNNState.cpp \
       $(SRC_DIR)/main.cpp

# Project headers
HEADERS := $(INCLUDES_DIR)/Machine.hpp \
		   $(INCLUDES_DIR)/json.hpp \
           $(CLASS_DIR)/ui/Button.hpp \
           $(CLASS_DIR)/NeuralNetwork.hpp \
           $(CLASS_DIR)/CNN.hpp \
           $(CLASS_DIR)/Perceptron.hpp \
           $(CLASS_DIR)/JsonParser.hpp \
           $(CLASS_DIR)/DMatrix.hpp

# Include paths
INC_FLAGS := -I$(CLASS_DIR) \
             -I$(INCLUDES_DIR) \
             -I$(IMGUI_DIR) \
             -I$(IMGUI_EXTRAS)

# Libraries
LIBS := $(INCLUDES_DIR)/raylib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11

# Object files with build directory
IMGUI_OBJS := $(IMGUI_SRC:%.cpp=$(OBJ_DIR)/%.o)
OBJS := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
ALL_OBJS := $(IMGUI_OBJS) $(OBJS)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

# Default target
.DEFAULT_GOAL := all

# Main targets
all: $(NAME)
	@echo "$(GREEN)✓ Build complete: $(NAME)$(NC)"

$(NAME): $(ALL_OBJS)
	@echo "$(YELLOW)Linking $(NAME)...$(NC)"
	@$(CXX) $(CXXFLAGS) $(ALL_OBJS) -o $@ $(LIBS)

# Compile project sources
$(OBJ_DIR)/%.o: %.cpp $(HEADERS)
	@echo "$(YELLOW)Compiling $<...$(NC)"
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

# Clean targets
clean:
	@echo "$(RED)Cleaning object files...$(NC)"
	@rm -rf $(BUILD_DIR)

fclean: clean
	@echo "$(RED)Cleaning executable...$(NC)"
	@rm -f $(NAME)

re: fclean all

# Utility targets
run: all
	@echo "$(GREEN)Running $(NAME)...$(NC)"
	@./$(NAME)

debug: all
	@echo "$(GREEN)Running $(NAME) with valgrind...$(NC)"
	@$(DEBUG) ./$(NAME)

# Show compilation commands (for debugging makefile)
verbose: CXXFLAGS += -v
verbose: re

# Help target
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

.PHONY: all clean fclean re run debug verbose
