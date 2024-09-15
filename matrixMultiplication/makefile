# Compiler and flags
CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
CXX             = g++
CXX_FLAGS       = -I$(CUDA_PATH)/include -Iinclude -g -O2
NVCC_FLAGS      = -I$(CUDA_PATH)/include -Iinclude -g -G -lineinfo
LDFLAGS         = -L$(CUDA_PATH)/lib64 -lcudart

# Directories
SRC_DIR         = src
INCLUDE_DIR     = include
BUILD_DIR       = build

# Source files
CU_SOURCES      = $(SRC_DIR)/main.cu
#CPP_SOURCES     = $(SRC_DIR)/main.cpp
CU_OBJECTS      = $(CU_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
#CPP_OBJECTS     = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Output executable
EXECUTABLE = matrix_multiplication

# Default target: build the executable
all: $(BUILD_DIR) $(EXECUTABLE)

# Build the executable
$(EXECUTABLE): $(CU_OBJECTS)
	echo $(CU_OBJECTS)
	$(NVCC) $(CU_OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

# Compile CUDA source files to object files
$(CU_OBJECTS): $(CU_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files to object files
# $(CPP_OBJECTS): $(CPP_SOURCES)
# 	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: all clean
