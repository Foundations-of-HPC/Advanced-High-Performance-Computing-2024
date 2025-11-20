CC = gcc
FC = pgf90
NVCC = nvcc

CFLAG = -O3
FFLAG = -Mcuda=cc70 -fast
NVCCFLAG = -arch=sm_70 -Xcompiler=-O3

dummy:
	@echo -e "make cuda\t\tcompile all CUDA sources"
	@echo -e "make cudafor\t\tcompile all CUDAFORTRAN sources"
	@echo -e "make clean\t\trm -f *.x *.mod"

cuda:    MMShared_c.x
cudafor: MMShared_f.x

MMShared_c.x: MMShared.cu
	$(NVCC) $(NVCCFLAG) $< -o $@

MMShared_f.x: MMShared.F90
	$(FC) $(FFLAG) $< -o $@

clean:
	rm -f *.x *.mod *.ptx
