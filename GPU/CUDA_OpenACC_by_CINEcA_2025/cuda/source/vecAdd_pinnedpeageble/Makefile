# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -g -G -O3 -arch=sm_80 -lnvToolsExt

# Pattern rule: compile .cu files to .x executables
%.x: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Default target
all: vecAdd_managed.x

# Clean up executables
clean:
	rm -f *.x

# Run the executable
run: vecAdd_managed.x
	./vecAdd_managed.x

.PHONY: all clean run
