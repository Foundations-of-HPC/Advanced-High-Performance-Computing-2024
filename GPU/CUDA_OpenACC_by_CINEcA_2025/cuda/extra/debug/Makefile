NVCC = nvcc
NVCC_FLAGS = -arch=sm_80 -lineinfo --ptxas-options=--verbose

SRCS := $(wildcard *.cu)
EXES  = $(patsubst %.cu,%.x,$(SRCS))

all: exe

exe : $(EXES)

help:
	@echo -e "make exe  \t compile: $(SRCS)"

%.x: %.cu
	@echo -e "\nCOMPILE: $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	$(RM) $(EXES) 

