PGCC  = pgcc -O3 -ta=tesla -Minfo=all -acc=noautopar
PGF90 = pgfortran -O3 -ta=tesla -Minfo=all -acc=noautopar

BIN = binary

help:
	@echo "Please specify the make target"
	@echo "make pgi_c or make pgi_f90"

pgi_c: *.c
	$(PGCC) -o $(BIN) $<

pgi_f90: *.F90
	$(PGF90) -o $(BIN) $<

clean:
	$(RM) $(BIN) 
