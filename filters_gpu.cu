#include "kernels.h"


/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m_gpu[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };

int8_t lp5_m_gpu[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };

/* Laplacian of gaussian */
int8_t log_m_gpu[] =
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

/* Identity */
int8_t identity_m_gpu[] = {1};

int8_t *builtin_filters_int[NUM_FILTERS] = {lp3_m_gpu, lp5_m_gpu, log_m_gpu, identity_m_gpu};





__device__ int32_t apply2d_gpu(const int8_t *f, int32_t dimension, const int32_t *original, int32_t *target,
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
                int32_t orig_pixel = original[target_r * width + target_c];
                new_pixel = new_pixel + orig_pixel * f[filter_r * dimension + filter_c];
            }
        }
    }

    return new_pixel;
}