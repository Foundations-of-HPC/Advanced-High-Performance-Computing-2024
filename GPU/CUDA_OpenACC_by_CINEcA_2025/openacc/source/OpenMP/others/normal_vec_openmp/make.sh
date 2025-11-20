# ----- 
# You can also try this 
# gcc -c -fopenmp -flto map_vec_add.c 
# gcc -c -fopenmp  map_vec_add.o

# gcc -c -fopenmp -flto map_vec_add.c && gcc -fopenmp map_vec_add.o

# gcc -fopenmp -c -O1 -fdse -foffload=-mavx map_vec_add.c
# gcc -fopenmp -foffload="-O3 -v" map_vec_add.o


gcc -fopenmp normal_vec_add.c 




