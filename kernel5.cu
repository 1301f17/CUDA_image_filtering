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

/* This is your own kernel, you should decide which parameters to add
   here*/

__device__ int32_t apply2d_gpu_texture(const int8_t *f, int32_t dimension, cudaTextureObject_t tex, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t new_pixel = 0;

    // Coordinate of the top left neighbour of the pixel.
    int top_left_r = row - (dimension - 1)/2;
    int top_left_c = column - (dimension - 1)/2;

    // Apply filter (sum the products one by one)
    for (int filter_r = 0; filter_r < dimension; filter_r++) {
        for (int filter_c = 0; filter_c < dimension; filter_c++) {
            int target_r = top_left_r + filter_r;
            int target_c = top_left_c + filter_c;
            if (target_r >= 0 && target_r < height && target_c >= 0 && target_c < width) {
                int32_t orig_pixel = tex2D<int>(tex, target_c, target_r);
                new_pixel = new_pixel + orig_pixel * f[filter_r * dimension + filter_c];
            }
        }
    }

    return new_pixel;
}

__global__ void kernel5(int32_t dimension, 
        cudaTextureObject_t tex, int32_t *output, int32_t width, int32_t height)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x; 
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (index_x < width && index_y < height) {
		output[index_y * width + index_x] = apply2d_gpu_texture(filter_constant, dimension, tex, output, width, height, index_y, index_x);
	}
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
        int32_t smallest, int32_t biggest)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x; 
    int index_y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (index_x < width && index_y < height && biggest != smallest) {
		image[index_y * width + index_x] = ((image[index_y * width + index_x] - smallest) * 255) / (biggest - smallest);
	}
}


__device__ int32_t apply2d_gpu_texture_1d(const int8_t *filter, int32_t dimension, cudaTextureObject_t tex, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t new_pixel = 0;
 
    // Coordinate of the top left neighbour of the pixel.
    int top_left_r = row - (dimension - 1)/2;
    int top_left_c = column - (dimension - 1)/2;
 
    // Apply filter (sum the products one by one)
    for (int filter_r = 0; filter_r < dimension; filter_r++) {
        for (int filter_c = 0; filter_c < dimension; filter_c++) {
            int target_r = top_left_r + filter_r;
            int target_c = top_left_c + filter_c;
            if (target_r >= 0 && target_r < height && target_c >= 0 && target_c < width) {
                int32_t orig_pixel = tex1Dfetch<int>(tex, target_r * width + target_c);
                new_pixel = new_pixel + orig_pixel * filter[filter_r * dimension + filter_c];
            }
        }
    }
 
    return new_pixel;
}
 
__global__ void kernel5_1d(int32_t dimension, 
        cudaTextureObject_t tex, int32_t *output, int32_t width, int32_t height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int row = index / width;
  int column = index % width;
  if (index < width * height) {
    output[index] = apply2d_gpu_texture_1d(filter_constant, dimension, tex, output, width, height, row, column);
  }
}
 
__global__ void normalize5_1d(int32_t *image, int32_t width, int32_t height,
        int32_t smallest, int32_t biggest)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < width * height && biggest != smallest) {
    image[index] = ((image[index] - smallest) * 255) / (biggest - smallest);
  }  
}