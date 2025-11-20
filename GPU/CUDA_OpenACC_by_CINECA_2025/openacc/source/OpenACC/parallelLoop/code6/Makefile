GCC = gcc
FOR = gfortran
NVCC  = nvc -O3 -ta=tesla -Minfo=all -acc=noautopar
NVF90 = nvfortran -O3 -ta=tesla -Minfo=all -acc=noautopar

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

