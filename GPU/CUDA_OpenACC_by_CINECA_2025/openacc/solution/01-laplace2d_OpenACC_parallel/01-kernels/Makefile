NVCC  = nvc -fast -acc -O3 -Minfo=all -acc=noautopar
NVF90 = nvfortran -fast -acc -O3 -Minfo=all -acc=noautopar

BIN = laplace2d_kernel.x

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
