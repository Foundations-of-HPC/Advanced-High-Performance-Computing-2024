PGCC  = nvc -O3 -gpu=cc80 -Minfo=all -acc=noautopar
PGF90 = nvfortran -O3 -gpu=cc80 -Minfo=all -acc=noautopar

NVCC = nvcc
NVCC_FLAGS = -arch=sm_80 -lineinfo --ptxas-options=--verbose -Xptxas -dlcm=ca
# -Xptxas -dlcm=ca
# -Xptxas -dlcm=cg # L2 only

DIMS = 555
ITER = 2

NAME := $(sort $(wildcard *.cu))
NAME := $(filter-out max.cu,$(NAME))
NAME := $(patsubst %.cu,%,$(NAME))

SRCS  = $(patsubst %,%.cu,$(NAME))
EXES  = $(patsubst %,%.x,$(NAME))
NCUS  = $(patsubst %,%.ncu-rep,$(NAME))
RUNS  = $(patsubst %,%.run,$(NAME))
METS  = $(patsubst %,%.metrics,$(NAME))
SANIT = $(patsubst %,%.sanitizer,$(NAME))

.PHONY: all clean cleanall run exe prof # $(NCUS)

all: exe

exe : $(EXES)
run : $(RUNS)
ncu : $(NCUS)
prof: $(METS)
sanitizer: $(SANIT)


help:
	@echo -e "make exe  \t compile: $(SRCS)"
	@echo -e "make run  \t run: $(EXES)"
	@echo -e "make ncu  \t profile: $(EXES)"
	@echo -e "make prof \t metrics: $(EXES)"

%.x: %.cu
	@echo -e "\nCOMPILE: $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

%.run: %.x
	@echo -e "\nRUN: $<"
	./$< $(DIMS)

%.ncu-rep: %.x
	@echo -e "\nNCU: $<"
	ncu --set full -f -k jacobi -o $(@:.ncu-rep=) ./$< $(DIMS) $(ITER)
	#ncu --cache-control all --set full -f -k jacobi -o $(@:.ncu-rep=) ./$< $(DIMS) $(ITER)

%.metrics: %.x
	@echo -e "\nMETRICS: $<"
	ncu --cache-control all -k jacobi --metrics $(METRICS) ./$< $(DIMS) 1

%.sanitizer: %.x
	@echo -e "\nSANITIZER: $<"
	compute-sanitizer ./$< $(DIMS) 1

clean:
	$(RM) $(EXES) 

cleanall: clean
	$(RM) $(NCUS)

