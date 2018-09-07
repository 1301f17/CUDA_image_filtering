CC=/usr/local/cuda-8.0/bin/nvcc

CAPABILITY=-arch=sm_50

GCC_OPT = $(CAPABILITY) -O2 -std=c++11 --compiler-options -Wall

OBJ  = pgm.o kernel1.o kernel2.o kernel3.o kernel4.o kernel5.o reduction.o filters_gpu.o filters.cu
DEPS = pgm.h kernels.h

%.o: %.cu $(DEPS)
	$(CC) --device-c $(GCC_OPT) -c -o $@ $<

all: $(OBJ) main.o
	$(CC) $(GCC_OPT) $^ -o solution.out

pgm_creator: pgm_creator.o pgm.o
	$(CC) $(GCC_OPT) $^ -o pgm_creator.out

difference: difference.o pgm.o
	$(CC) $(GCC_OPT) $^ -o difference

clean:
	rm -f *.o *.out
