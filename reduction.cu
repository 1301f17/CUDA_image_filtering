#include "kernels.h"


__inline__ __device__
int warpReduceMax(int val)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
    {
        val = max(val, __shfl_down(val, offset));
    }
    return val;
}

__inline__ __device__
int warpReduceMin(int val)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
    {
        val = min(val, __shfl_down(val, offset));
    }
    return val;
}


__global__ void reduction(int32_t *max_image, int32_t *min_image, int32_t N,
        int32_t *output_max, int32_t *output_min)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max = index < N? max_image[index]:-99999; // give the threads with no pixel a small value so that it will not influence the max
    int min = index < N? min_image[index]:99999;

    static __shared__ int shared_max[32]; // Shared mem for 32 partial sums
    static __shared__ int shared_min[32]; // Shared mem for 32 partial sums

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    max = warpReduceMax(max); 
    min = warpReduceMin(min);     // Each warp performs partial reduction

    if (lane==0) {
        shared_max[wid]=max;
        shared_min[wid]=min;
    } // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    max = (threadIdx.x < blockDim.x / warpSize) ? shared_max[lane] : -99999;
    min = (threadIdx.x < blockDim.x / warpSize) ? shared_min[lane] : 99999;

    if (wid==0) {
        max = warpReduceMax(max); //Final reduce within first warp
        min = warpReduceMin(min); //Final reduce within first warp
    }

    if (threadIdx.x==0) {
        output_max[blockIdx.x] = max;
        output_min[blockIdx.x] = min;
    }   
}
