# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -O2 -std=c++11

# Source and output
SRC = matrixMultiplication.cpp
OUT = mmul.x

# Default target
all: $(OUT)

# Rule to compile the CUDA C++ code
$(OUT): $(SRC)
	$(NVCC) -ccbin=$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)

# Clean up the build
clean:
	rm -f $(OUT)

# Phony targets (for non-file targets)
.PHONY: all clean

