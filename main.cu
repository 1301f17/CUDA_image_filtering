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

#include <stdio.h>
#include <string>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <time.h>

#include "pgm.h"
#include "filters.h"
#include "kernels.h"
#include "clock.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
        float time_gpu_transfer_in, float time_gpu_transfer_out)
{
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu/time_gpu_computation);
    printf("%7.2f\n", time_cpu/
            (time_gpu_computation  + time_gpu_transfer_in + time_gpu_transfer_out));
}

__constant__ int8_t filter_constant[9*9];
int filter_dimension = 9;
int filter_index = 2;

int main(int argc, char **argv)
{
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3)
    {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt (argc, argv, "i:o:")) != -1)
    {
        switch (c)
        {
            case 'i':
                input_filename = std::string(optarg);
                break;
            case 'o':
                cpu_output_filename = std::string(optarg);
                base_gpu_output_filename = std::string(optarg);
                break;
            default:
                return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
    {
       printf("Error loading source image.\n");
       return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
            "Speedup_noTrf Speedup\n");

    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img); 

    float time_cpu;
    struct timespec start, stop;

    /*
     * TODO: Run your CPU implementation here and get its time. Don't include
     * file IO in your measurement.
     * */

    clock_gettime(CLOCK_MONOTONIC, &start); 
    apply_filter2d_threaded(builtin_filters[filter_index],
            source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height,
            get_nprocs());
    clock_gettime(CLOCK_MONOTONIC, &stop);
    time_cpu = (stop.tv_sec - start.tv_sec)*1000 + (double)(stop.tv_nsec - start.tv_nsec) / 1000000;

    save_pgm_to_file(cpu_output_filename.c_str(), &cpu_output_img);

    /*
     * CPU implementation ends
     * */

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);

    // create source image at host and device
    int32_t n = source_img.width * source_img.height;
    int32_t *d_source_img;
    cudaMalloc((void **)&d_source_img, n*sizeof(int));


    int32_t threads_per_block = min(1024, properties.maxThreadsPerBlock);

    int8_t *filter;
    cudaMalloc((void **)&filter, filter_dimension*filter_dimension*sizeof(int8_t));
    cudaMemcpy(filter, builtin_filters_int[filter_index], filter_dimension*filter_dimension*sizeof(int8_t), cudaMemcpyHostToDevice); 

    cudaMemcpyToSymbol(filter_constant, builtin_filters_int[2], 9*9*sizeof(int8_t));  

    int32_t blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block; // number of blocks we need in reduction
    int num_iterations = 1; // number of times we need to call the reduction kernels to get the min/max
    int temp = blocks_reduction;
    while (temp > 1) {
        num_iterations += 1;
        temp = (temp + (threads_per_block - 1)) / threads_per_block;
    }


    int32_t *d_max; // device memory for the max values
    int32_t *d_min; // device memory for the min values
    cudaMalloc((void **)&d_max, blocks_reduction * sizeof(int));
    cudaMalloc((void **)&d_min, blocks_reduction * sizeof(int));

    int32_t n_reduction; // total number of values we want to put into the reduction kernel
    int host_max = 255;
    int host_min = 0;

    Clock gpu_clock;
    float time_in, time_kernel, time_out;


    /* TODO:
     * run each of your gpu implementations here,
     * get their time,
     * and save the output image to a file.
     * Don't forget to add the number of the kernel
     * as a prefix to the output filename:
     * Print the execution times by calling print_run().
     */

    /*
     * Kernel 1
     * */

    // create some basic information
    std::string gpu_file1 = "1"+base_gpu_output_filename;
    int32_t blocks_1 = (n + (threads_per_block - 1)) / threads_per_block;
    blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block;

    // transfer in 
    gpu_clock.start();
    cudaMemcpy(d_source_img, source_img.matrix, n*sizeof(int), cudaMemcpyHostToDevice);
    time_in = gpu_clock.stop() * 1000;

    // create output image at host and device
    pgm_image gpu_output_img1;
    copy_pgm_image_size(&source_img, &gpu_output_img1);
    int32_t *d_output_img1;
    cudaMalloc((void **)&d_output_img1, n*sizeof(int));

    // call kernel1
    gpu_clock.start();
    kernel1<<< blocks_1, threads_per_block >>>(filter, filter_dimension, d_source_img, d_output_img1, source_img.width, source_img.height);

    /* reduction to calculate max and min */
    n_reduction = n;
    reduction<<< blocks_reduction, threads_per_block >>>(d_output_img1, d_output_img1, n_reduction, d_max, d_min);
    n_reduction = blocks_reduction;
    blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < num_iterations - 1; i++) {
        reduction<<< blocks_reduction, threads_per_block >>>(d_max, d_min, n_reduction, d_max, d_min);
        n_reduction = blocks_reduction;
        blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;
    }

    // copy min and max back
    cudaMemcpy(&host_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d %d", host_max, host_min);

    // normalize
    normalize1<<< blocks_1, threads_per_block >>>(d_output_img1, source_img.width, source_img.height, host_min, host_max);

    time_kernel = gpu_clock.stop() * 1000;

    // copy back to host
    gpu_clock.start();
    cudaMemcpy(gpu_output_img1.matrix, d_output_img1, n * sizeof(int), cudaMemcpyDeviceToHost);
    time_out = gpu_clock.stop() * 1000;

    print_run(time_cpu, 1, time_kernel, time_in, time_out);
    save_pgm_to_file(gpu_file1.c_str(), &gpu_output_img1);

    // Free memory
    cudaFree(d_output_img1);
    destroy_pgm_image(&gpu_output_img1);
    /*
     * Kernel 1 end
     * */


    /*
     * Kernel 2
     * */
    // create some basic information
    std::string gpu_file2 = "2"+base_gpu_output_filename;
    int32_t blocks_2 = (n + (threads_per_block - 1)) / threads_per_block;
    blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block;

    // transfer in
    gpu_clock.start();
    cudaMemcpy(d_source_img, source_img.matrix, n*sizeof(int), cudaMemcpyHostToDevice);
    time_in = gpu_clock.stop() * 1000;

    // create output image at host and device
    pgm_image gpu_output_img2;
    copy_pgm_image_size(&source_img, &gpu_output_img2);
    int32_t *d_output_img2;
    cudaMalloc((void **)&d_output_img2, n*sizeof(int));

    // call kernel
    gpu_clock.start();
    kernel2<<< blocks_2, threads_per_block >>>(filter, filter_dimension, d_source_img, d_output_img2, source_img.width, source_img.height);

    /* reduction to calculate max and min */

    n_reduction = n;
    reduction<<< blocks_reduction, threads_per_block >>>(d_output_img2, d_output_img2, n_reduction, d_max, d_min);
    n_reduction = blocks_reduction;
    blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < num_iterations - 1; i++) {
        reduction<<< blocks_reduction, threads_per_block >>>(d_max, d_min, n_reduction, d_max, d_min);
        n_reduction = blocks_reduction;
        blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;
    }

    // copy min and max back
    cudaMemcpy(&host_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d %d", host_max, host_min);

    // normalize
    normalize2<<< blocks_2, threads_per_block >>>(d_output_img2, source_img.width, source_img.height, host_min, host_max);

    time_kernel = gpu_clock.stop() * 1000;

    // copy back to host
    gpu_clock.start();
    cudaMemcpy(gpu_output_img2.matrix, d_output_img2, n * sizeof(int), cudaMemcpyDeviceToHost);
    time_out = gpu_clock.stop() * 1000;

    print_run(time_cpu, 2, time_kernel, time_in, time_out);
    save_pgm_to_file(gpu_file2.c_str(), &gpu_output_img2);

    // Free memory
    cudaFree(d_output_img2);
    destroy_pgm_image(&gpu_output_img2);
    /*
     * Kernel 2 end
     * */


    /*
     * Kernel 3
     * */
    // create some basic information
    std::string gpu_file3 = "3"+base_gpu_output_filename;

    int nrows = 1;
    int32_t blocks_3 = ((source_img.height + nrows - 1)/nrows + (threads_per_block - 1)) / threads_per_block;
    blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block;

    // transfer in
    gpu_clock.start();
    cudaMemcpy(d_source_img, source_img.matrix, n*sizeof(int), cudaMemcpyHostToDevice);
    time_in = gpu_clock.stop() * 1000;

    // create output image at host and device
    pgm_image gpu_output_img3;
    copy_pgm_image_size(&source_img, &gpu_output_img3);
    int32_t *d_output_img3;
    cudaMalloc((void **)&d_output_img3, n*sizeof(int));

    // call kernel
    gpu_clock.start();
    kernel3<<< blocks_3, threads_per_block >>>(filter, filter_dimension, d_source_img, d_output_img3, source_img.width, source_img.height, nrows);

    /* reduction to calculate max and min */

    n_reduction = n;
    reduction<<< blocks_reduction, threads_per_block >>>(d_output_img3, d_output_img3, n_reduction, d_max, d_min);    
    n_reduction = blocks_reduction;
    blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < num_iterations - 1; i++) {
        reduction<<< blocks_reduction, threads_per_block >>>(d_max, d_min, n_reduction, d_max, d_min);
        n_reduction = blocks_reduction;
        blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;
    }

    // copy min and max back
    cudaMemcpy(&host_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d %d", host_max, host_min);

    // normalize
    normalize3<<< blocks_3, threads_per_block >>>(d_output_img3, source_img.width, source_img.height, host_min, host_max, nrows);

    time_kernel = gpu_clock.stop() * 1000;

    // copy back to host
    gpu_clock.start();
    cudaMemcpy(gpu_output_img3.matrix, d_output_img3, n * sizeof(int), cudaMemcpyDeviceToHost);
    time_out = gpu_clock.stop() * 1000;

    print_run(time_cpu, 3, time_kernel, time_in, time_out);
    save_pgm_to_file(gpu_file3.c_str(), &gpu_output_img3);

    // Free memory
    cudaFree(d_output_img3);
    destroy_pgm_image(&gpu_output_img3);
    /*
    * Kernel 3 end
    * */


    /*
     * Kernel 4
     * */
    // create some basic information
    std::string gpu_file4 = "4"+base_gpu_output_filename;
    int pixels_per_thread = 8;
    int32_t blocks_4 = ((n + pixels_per_thread - 1)/pixels_per_thread + (threads_per_block - 1)) / threads_per_block;
    blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block;

    // transfer in
    gpu_clock.start();
    cudaMemcpy(d_source_img, source_img.matrix, n*sizeof(int), cudaMemcpyHostToDevice);
    time_in = gpu_clock.stop() * 1000;

    // create output image at host and device
    pgm_image gpu_output_img4;
    copy_pgm_image_size(&source_img, &gpu_output_img4);
    int32_t *d_output_img4;
    cudaMalloc((void **)&d_output_img4, n*sizeof(int));

    // call kernel
    gpu_clock.start();
    kernel4<<< blocks_4, threads_per_block >>>(filter, filter_dimension, d_source_img, d_output_img4, source_img.width, source_img.height);

    /* reduction to calculate max and min */

    n_reduction = n;
    reduction<<< blocks_reduction, threads_per_block >>>(d_output_img4, d_output_img4, n_reduction, d_max, d_min);    
    n_reduction = blocks_reduction;
    blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < num_iterations - 1; i++) {
        reduction<<< blocks_reduction, threads_per_block >>>(d_max, d_min, n_reduction, d_max, d_min);
        n_reduction = blocks_reduction;
        blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;
    }

    // copy min and max back
    cudaMemcpy(&host_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d %d", host_max, host_min);

    // normalize
    normalize4<<< blocks_4, threads_per_block >>>(d_output_img4, source_img.width, source_img.height, host_min, host_max);

    time_kernel = gpu_clock.stop() * 1000;

    // copy back to host
    gpu_clock.start();
    cudaMemcpy(gpu_output_img4.matrix, d_output_img4, n * sizeof(int), cudaMemcpyDeviceToHost);
    time_out = gpu_clock.stop() * 1000;

    print_run(time_cpu, 4, time_kernel, time_in, time_out);
    save_pgm_to_file(gpu_file4.c_str(), &gpu_output_img4);

    // Free memory
    cudaFree(d_output_img4);
    destroy_pgm_image(&gpu_output_img4);
    /*
    * Kernel 4 end
    * */


    /*
     * Kernel 5
     * */
    std::string gpu_file5 = "5"+base_gpu_output_filename;
    int32_t blocks_5 = (n + (threads_per_block - 1)) / threads_per_block;
    blocks_reduction = (n + (threads_per_block - 1)) / threads_per_block;

    int32_t *d_source_img2D;

    bool _2d = false;
    // 2D Pitch restriction
    if (source_img.width < 60000 && source_img.height < 60000) {
        _2d = true;
    }

    // transfer in
    gpu_clock.start();

    // Prepare Resource/Texture objects
    cudaTextureObject_t tex;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    if (_2d) {
        // Use 2D Pitch Resource
        // Reference: https://stackoverflow.com/questions/16380883/new-cuda-texture-object-getting-wrong-data-in-2d-case
        size_t pitch; 
        cudaMallocPitch(&d_source_img2D, &pitch, sizeof(int) * source_img.width, source_img.height);
        cudaMemcpy2D(d_source_img2D, pitch, source_img.matrix, sizeof(int) * source_img.width,
                     sizeof(int) * source_img.width, source_img.height, cudaMemcpyHostToDevice);
        time_in = gpu_clock.stop() * 1000;

        // Specify Resource details
        resDesc.resType = cudaResourceTypePitch2D; // for 2D texture
        resDesc.res.pitch2D.devPtr = d_source_img2D;
        resDesc.res.pitch2D.pitchInBytes =  pitch; 
        resDesc.res.pitch2D.width = source_img.width; 
        resDesc.res.pitch2D.height = source_img.height; 
        resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat; 
        resDesc.res.pitch2D.desc.x = 32; // bits per channel 
        resDesc.res.pitch2D.desc.y = 0; 

    } else {
        cudaMemcpy(d_source_img, source_img.matrix, n*sizeof(int), cudaMemcpyHostToDevice);
        time_in = gpu_clock.stop() * 1000;

        // Specify Resource details
        // Reference: https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
        resDesc.resType = cudaResourceTypeLinear; // for 1D texture
        resDesc.res.linear.devPtr = d_source_img;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = n*sizeof(int);
     
    }
    // Create texture object
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    // create output image at host and device
    pgm_image gpu_output_img5;
    copy_pgm_image_size(&source_img, &gpu_output_img5);
    int32_t *d_output_img5;
    cudaMalloc((void **)&d_output_img5, n*sizeof(int));

    dim3 gridSize((source_img.width + 31) / 32, (source_img.height + 31) / 32);
    dim3 blockSize(32, 32);

    // call kernel
    gpu_clock.start();
    if (_2d) {
        kernel5<<< gridSize, blockSize >>>(filter_dimension, tex, d_output_img5, source_img.width, source_img.height);
    } else {
        kernel5_1d<<< blocks_5, threads_per_block >>>(filter_dimension, tex, d_output_img5, source_img.width, source_img.height);
    }

    /* reduction to calculate max and min */

    n_reduction = n;
    reduction<<< blocks_reduction, threads_per_block >>>(d_output_img5, d_output_img5, n_reduction, d_max, d_min);    
    n_reduction = blocks_reduction;
    blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < num_iterations - 1; i++) {
        reduction<<< blocks_reduction, threads_per_block >>>(d_max, d_min, n_reduction, d_max, d_min);
        n_reduction = blocks_reduction;
        blocks_reduction = (blocks_reduction + (threads_per_block - 1)) / threads_per_block;
    }

    // copy min and max back
    cudaMemcpy(&host_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d %d", host_max, host_min);

    // normalize
    if (_2d) {
        normalize5<<< gridSize, blockSize >>>(d_output_img5, source_img.width, source_img.height, host_min, host_max);
    } else {
        normalize5_1d<<< blocks_5, threads_per_block >>>(d_output_img5, source_img.width, source_img.height, host_min, host_max);
    }

    time_kernel = gpu_clock.stop() * 1000;

    // copy back to host
    gpu_clock.start();
    cudaMemcpy(gpu_output_img5.matrix, d_output_img5, n * sizeof(int), cudaMemcpyDeviceToHost);
    time_out = gpu_clock.stop() * 1000;

    print_run(time_cpu, 5, time_kernel, time_in, time_out);
    save_pgm_to_file(gpu_file5.c_str(), &gpu_output_img5);

    // Free memory
    cudaFree(d_source_img2D);
    cudaFree(d_source_img);
    cudaFree(d_output_img5);
    destroy_pgm_image(&gpu_output_img5);
    /*
    * Kernel 5 end
    * */

    // Free memory
    cudaDestroyTextureObject(tex);
    cudaFree(filter);
    cudaFree(d_max);
    cudaFree(d_min);
}
