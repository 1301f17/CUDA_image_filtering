/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367H1 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2017 Bogdan Simion
 * -------------
*/

#include "kernels.h"


__global__ void kernel1(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index % height;
	int column = index / height;
	int image_index = row * width + column;
	if (index < width * height) {
		output[image_index] = apply2d_gpu(filter, dimension, input, output, width, height, row, column);
	}
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
        int32_t smallest, int32_t biggest)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index % height;
	int column = index / height;
	int image_index = row * width + column;
	if (index < width * height && biggest != smallest) {
		image[image_index] = ((image[image_index] - smallest) * 255) / (biggest - smallest);
	}	
}
