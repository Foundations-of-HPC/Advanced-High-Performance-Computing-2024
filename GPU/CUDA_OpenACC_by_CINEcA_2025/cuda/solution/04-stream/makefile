CXX=nvcc
CXXFLAGS = -arch=sm_80

SOURCES = stream0 stream1

%: %.cu
	$(CXX) $(CXXFLAGS) -o $@.out $^
ifneq (,$(findstring r,  $(MAKEFLAGS)))
	./$@.out
endif

all: $(SOURCES)

.PHONY: clean

clean:
	find . -name "*.out" -delete


