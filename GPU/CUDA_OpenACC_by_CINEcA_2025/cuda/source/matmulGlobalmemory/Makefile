# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -g -G -O3 -arch=sm_80 -lnvToolsExt

# Source and target files
CU_SRCS := $(wildcard *.cu)
EXECUTABLES := $(CU_SRCS:.cu=.x)

# Default target
all: $(EXECUTABLES)

# Pattern rule to compile .cu files to .x executables
%.x: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Clean up executables
clean:
	rm -f *.x

# Run the first executable as an example
run: $(EXECUTABLES)
	./$(word 1, $^)

.PHONY: all clean run

