# Compiler and flags
CXX = nvc++
CXXFLAGS = -O2 -std=c++11 -cuda

# Source and output
SRC = single-block-loop.cu
OUT = bin

# Default target
all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean

