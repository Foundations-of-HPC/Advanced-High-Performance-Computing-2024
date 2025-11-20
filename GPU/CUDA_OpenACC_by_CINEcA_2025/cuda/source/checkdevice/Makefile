# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -g -G -O3 -arch=sm_80

# Source files
SOURCES = warp.cu 

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Executable name
EXECUTABLE = bin.x

# Default target
all: $(EXECUTABLE)

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link object files to create executable
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@

# Clean up object files and executable
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: all clean run

