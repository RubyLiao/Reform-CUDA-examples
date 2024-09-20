CXX             = g++
CXX_FLAGS       = -I/Users/liaoyu/Documents/DL/Reform-CUDA-examples/tryInclude/include -g -O2

SRC_DIR         = src
INCLUDE_DIR     = include
BUILD_DIR       = build


SOURCES  = $(SRC_DIR)/main.cpp $(SRC_DIR)/a.cpp $(SRC_DIR)/b.cpp $(SRC_DIR)/c.cpp
OBJECTS  = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)



EXECUTABLE = tryInclude

# Default target: build the executable
all: $(BUILD_DIR) $(EXECUTABLE)

# Build the executable
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXECUTABLE)

# Compile CUDA source files to object files
$(OBJECTS): $(SOURCES)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: all clean