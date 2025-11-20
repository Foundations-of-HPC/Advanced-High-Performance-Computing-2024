NVCC  = nvc -mp -acc -O3 -Minfo=all -acc=noautopar
NVF90 = nvfortran -mp -acc -O3 -Minfo=all -acc=noautopar

BIN = laplace2d_parallel.x

help:
	@echo "Load Nvidia module: module load nvhpc"
	@echo "====================================="
	@echo "Please specify the make target"
	@echo "nvc or make nvf"

nvc: *.c
	$(NVCC) -o $(BIN) $<

nvf: *.F90
	$(NVF90) -o $(BIN) $<
clean:
	$(RM) $(BIN)
