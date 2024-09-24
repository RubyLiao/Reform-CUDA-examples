CXX             = g++
CXX_FLAGS       = -I/root/LY/Reform-CUDA-examples/tryInclude/include -g -O2

SRC_DIR         = src
INCLUDE_DIR     = include
BUILD_DIR       = build


# SOURCES  = $(SRC_DIR)/main.cpp $(SRC_DIR)/a.cpp $(SRC_DIR)/b.cpp $(SRC_DIR)/c.cpp
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)  # 获取所有源文件
OBJECTS  = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)



EXECUTABLE = tryInclude

# Default target: build the executable
all: $(BUILD_DIR) $(EXECUTABLE)

# $(EXECUTABLE): $(SOURCES)
# 	$(CXX) -o $@ $^ $(CXX_FLAGS)

# Build the executable
$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o $@ $^ $(CXX_FLAGS)

# Compile CUDA source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $< -o $@ $(CXX_FLAGS)

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: all clean