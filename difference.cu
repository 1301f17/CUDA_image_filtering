#include <stdio.h>
#include "pgm.h"

int main(int argc, char **argv)
{
	if (argc != 3) {
		printf("usage: difference filea fileb");
		return 0;
	}
    pgm_image imagea;
    pgm_image imageb;
    load_pgm_from_file(argv[1], &imagea);
    load_pgm_from_file(argv[2], &imageb);
    if (imagea.height != imageb.height || imagea.width != imageb.width) {
    	printf("dimension not matching!");
    	return 0;
    }
    for (int i=0; i<imagea.height*imagea.width; i++) {
    	if (imagea.matrix[i] != imageb.matrix[i]) {
    		printf("diff at index %d, %d : %d\n", i, imagea.matrix[i], imageb.matrix[i]);
    	}
    }
    return 0;
}