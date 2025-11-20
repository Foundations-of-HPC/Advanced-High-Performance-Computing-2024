GCC = gcc -fopenmp
FOR = gfortran -fopenmp
NVCC  = nvc -mp=gpu -Minfo=mp
NVF90 = nvfortran -mp=gpu -Minfo=mp

BIN = binary

help:
	@echo "Please specify the make target"
	@echo "make gcc or gfor nvc or make nvf"

nvc: *.c
	$(NVCC) -o $(BIN) $<

nvf: *.F90
	$(NVF90) -o $(BIN) $<
gcc: *.c
	$(GCC) -o $(BIN) $<
gfor: *.f90
	$(FOR) -o $(BIN) $<
clean:
	$(RM) $(BIN) 

