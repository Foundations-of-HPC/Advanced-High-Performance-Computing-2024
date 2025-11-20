# Compiler settings
NVCC = nvcc
ARCH = -arch=sm_80 # Adjust the architecture version to match your GPU
CXXFLAGS = -O2 -std=c++11

# Source and output files
SRC = gpuArray.cu
OUT = gpuArray.x

# Default target
all: $(OUT)

# Rule to compile the CUDA C++ code
$(OUT): $(SRC)
	$(NVCC) $(CXXFLAGS) $(ARCH) $(SRC) -o $(OUT)

# Clean up the build files
clean:
	rm -f $(OUT)

# Phony targets
.PHONY: all clean

