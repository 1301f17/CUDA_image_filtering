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

#include "filters.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_MIN 10000
#define DEFAULT_MAX -10000

typedef struct common_work_t
{
    parallel_method method;
    const filter *f;
    const int32_t *original;
    int32_t *target;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;

    // global variables for max/min elements
    pthread_mutex_t lock;
    int max_elem;
    int min_elem;
} common_work;

// For work pool method
typedef struct task_t
{
    int32_t row;
    int32_t col;
    int32_t work_chunk;
} task;

typedef struct task_queue_t
{
    task **tasks;
    int total_tasks;
    int current; // index of current task to do
    pthread_mutex_t current_lock;    
} task_queue;

typedef struct work_t
{
    common_work *common;
    task_queue *queue;
    int32_t id;
} work;

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
__attribute__ ((noinline)) void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }

    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t new_pixel = 0;

    // Coordinate of the top left neighbour of the pixel.
    int top_left_r = row - (f->dimension - 1)/2;
    int top_left_c = column - (f->dimension - 1)/2;

    // Apply filter (sum the products one by one)
    for (int filter_r = 0; filter_r < f->dimension; filter_r++) {
        for (int filter_c = 0; filter_c < f->dimension; filter_c++) {
            int target_r = top_left_r + filter_r;
            int target_c = top_left_c + filter_c;
            if (target_r >= 0 && target_r < height && target_c >= 0 && target_c < width) {
                int32_t orig_pixel = original[target_r * width + target_c];
                new_pixel += orig_pixel * f->matrix[filter_r * f->dimension + filter_c];
            }
        }
    }

    return new_pixel;
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
/* TODO: your sequential implementation goes here.
 */
void apply_filter2d(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
    int max = DEFAULT_MAX;
    int min = DEFAULT_MIN;

    for (int i = 0; i < width * height; i++) {
        int32_t new_pixel = apply2d(f, original, target, width, height, i / width, i % width);
        target[i] = new_pixel;

        // Tracks max and min
        if (new_pixel > max) {
            max = new_pixel;
        }
        if (new_pixel < min) {
            min = new_pixel;
        }
    }

    // Normalize all the pixels
    for (int j = 0; j < width * height; j++) {
        normalize_pixel(target, j, min, max);
    }
}

/****************** ROW/COLUMN SHARDING ************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */

/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void* sharding_work(void *args)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    work *wk = (work *)args;
    common_work *common = wk->common;
    int32_t id = wk->id;


    // Find the range of pixels this thread is responsible for and apply filter.
    int32_t num_rows = (common->height + common->max_threads - 1) / common->max_threads;
    int32_t num_pixels = num_rows * common->width;
    int32_t first_pixel = id * num_pixels;
    int32_t last_pixel = first_pixel + num_pixels - 1;
    if (last_pixel >= common->width * common->height) {
        last_pixel = common->width * common->height - 1;
    }

    // Track max and min
    int max = DEFAULT_MAX;
    int min = DEFAULT_MIN;

    // Apply filter
    for (int pix_i = first_pixel; pix_i <= last_pixel; pix_i++) {
        int32_t new_pixel = apply2d(common->f, common->original, common->target,
                                    common->width, common->height, pix_i / common->width, pix_i % common->width);
        common->target[pix_i] = new_pixel;

        // Tracks max and min
        if (new_pixel > max) {
            max = new_pixel;
        }
        if (new_pixel < min) {
            min = new_pixel;
        }
    }

    // Update local max/min to global location
    pthread_mutex_lock(&(common->lock));
    if (common->min_elem > min) {
        common->min_elem = min;
    }
    if (common->max_elem < max){
        common->max_elem = max;
    }
    pthread_mutex_unlock(&(common->lock));

    // Wait for all threads to complete the above tasks
    pthread_barrier_wait(&(common->barrier));

    // Normalize all the pixels
    for (int pix_i = first_pixel; pix_i <= last_pixel; pix_i++) {
        normalize_pixel(common->target, pix_i, common->min_elem, common->max_elem);
    }


    return NULL;
}



/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void apply_filter2d_threaded(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads)
{
    /* You probably want to define a struct to be passed as work for the
     * threads.
     * Some values are used by all threads, while others (like thread id)
     * are exclusive to a given thread. For instance:
     * (Check the top of the file for our implementation)
     *
     * An uglier (but simpler) solution is to define the shared variables
     * as global variables.
     */

    // Initialize common_work to share among the threads
    common_work *common = (common_work *) malloc(sizeof(common_work));

    common->f = f;
    common->original = original;
    common->target = target;
    common->width = width;
    common->height = height;
    common->max_threads = num_threads;
    pthread_barrier_init(&(common->barrier), NULL, num_threads);

    pthread_mutex_init(&(common->lock), NULL);
    common->max_elem = DEFAULT_MAX;
    common->min_elem = DEFAULT_MIN;

    // Spawn and join threads
    pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
 
    // Do sharding_work
    for(int i = 0; i < num_threads; i++) {
        // Set up work for each thread
        work *wk = (work *) malloc(sizeof(work));
        wk->id = i;
        wk->common = common;

        pthread_create(&threads[i], NULL, sharding_work, (void*)wk);
    }

    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    

}
