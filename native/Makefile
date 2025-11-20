# Makefile for native multiplication implementation
CXX = g++
CXXFLAGS = -std=c++23 -O3 -march=native -g
LDFLAGS = 

# Directories
UTILS_DIR = ../utils
BUILD_DIR = ../build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
SRC_DIR = ./src
INCLUDE_DIR = ./include

# Source files
UTILS_SRC = $(wildcard $(UTILS_DIR)/src/*.cpp)
NATIVE_SRC = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
UTILS_OBJ = $(patsubst $(UTILS_DIR)/src/%.cpp,$(OBJ_DIR)/%.o,$(UTILS_SRC))
NATIVE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(NATIVE_SRC))

# All object files
ALL_OBJ = $(UTILS_OBJ) $(NATIVE_OBJ)

# Output binary
BINARY = $(BIN_DIR)/native

# Include paths
INCLUDES = -I$(UTILS_DIR)/include -I$(INCLUDE_DIR)

# Phony targets
.PHONY: all utils native clean help

# Default target
all: $(BINARY)

# Build the main binary
$(BINARY): $(ALL_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(ALL_OBJ) -o $@ $(LDFLAGS)
	@echo "Built: $@"

# Build object files from utils sources
$(OBJ_DIR)/%.o: $(UTILS_DIR)/src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build object files from native sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@

# Utility targets
utils: $(UTILS_OBJ)
	@echo "Built utils objects"

native: $(NATIVE_OBJ)
	@echo "Built native objects"

# Clean build artifacts
clean:
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  utils   - Build utils library"
	@echo "  native  - Build native executable"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"
