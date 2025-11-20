# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++23 -O3 -march=native -g
LDFLAGS = 

# Directories
UTILS_DIR = ../utils
NATIVE_DIR = ../native
BUILD_DIR = .
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Source files
UTILS_SRC = $(wildcard $(UTILS_DIR)/src/*.cpp)
NATIVE_SRC = $(NATIVE_DIR)/native.cpp

# Object files
UTILS_OBJ = $(patsubst $(UTILS_DIR)/src/%.cpp,$(OBJ_DIR)/%.o,$(UTILS_SRC))
NATIVE_OBJ = $(patsubst $(NATIVE_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(NATIVE_SRC))

# All object files
ALL_OBJ = $(UTILS_OBJ) $(NATIVE_OBJ)

# Output binary
BINARY = $(BIN_DIR)/native

# Include paths
INCLUDES = -I$(UTILS_DIR)/include

# Phony targets
.PHONY: all utils native clean help

# Default target
all: $(BINARY)

# Build the main binary
$(BINARY): $(ALL_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(ALL_OBJ) -o $@ $(LDFLAGS)
	@echo "Built: $@"

# Build object files from utils and native sources
$(OBJ_DIR)/%.o: $(UTILS_DIR)/src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(NATIVE_DIR)/%.cpp | $(OBJ_DIR)
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