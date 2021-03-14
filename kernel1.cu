
 #include "kernels.h"
 #include <stdio.h>
 #define threads_block 512
 
 
 void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
  // Calculate threads and blocks.
  int num_threads, num_blocks;
  if (width * height > threads_block) {
    num_threads = threads_block;
    num_blocks = (width * height + threads_block - 1) / threads_block;
  }else {
    num_threads = width * height;
    num_blocks = 1;
  }
 
  dim3 dimBlock(num_threads, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);
  kernel1<<<dimGrid, dimBlock>>>(filter, dimension, input, output, width, height);
  // Initialiaze golbal device variables to store minimum and maximum.
  int32_t *gmin = NULL;
  int32_t *gmax = NULL;
  cudaMalloc((void **)&gmin, width * height * sizeof(int32_t));
  cudaMalloc((void **)&gmax, width * height * sizeof(int32_t));
  cudaMemcpy(gmin, output, width * height * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(gmax, output, width * height * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  int shMemSize = (num_threads <= 32) ? 4 * num_threads * sizeof(int32_t) : 2* num_threads * sizeof(int32_t);
  reduction<<<dimGrid, dimBlock, shMemSize>>>(gmin, gmax, width * height);

  // Repeat calling reduction kernel to get final maximum and minimum.
  while (num_blocks != 1) {
    int n = num_blocks;
    if (num_blocks > threads_block) {
      num_threads = threads_block;
      num_blocks = (num_blocks + threads_block - 1) / threads_block;
    }else {
      num_threads = num_blocks;
      num_blocks = 1;
    }
    shMemSize = (num_threads <= 32) ? 4 * num_threads * sizeof(int32_t) : 2* num_threads * sizeof(int32_t);
    dim3 newblock(num_threads, 1, 1);
    dim3 newgrid(num_blocks, 1, 1);
    reduction<<<newgrid, newblock, shMemSize>>>(gmin, gmax, n);
  }
  // Calling normalize kernel.
  normalize1<<<dimGrid, dimBlock>>>(output, width, height, gmin, gmax);
  cudaFree(gmin);
  cudaFree(gmax);
}
 
 __global__ void kernel1(const int8_t *filter, int32_t dimension,
                         const int32_t *input, int32_t *output, int32_t width,
                         int32_t height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < width * height) {
    idx = (idx % height) * width + idx / height;
    int32_t j;
    int counter = 0;
    output[idx] = 0;
    for (int k = 0; k < 5; k++) {
      for (j = idx - (2-k)*width - 2; j < idx - (2-k)*width + 3; j++) {
        if ((j >= 0 && j < width*height) && ((idx/width)-(j/width)+k == 2)) {
          output[idx] += filter[counter] * input[j]; 
        }
        counter++;
      }
    }
  }

}

__global__ void reduction(int32_t *smallest, int32_t *biggest, int n) {
  extern __shared__ int32_t array[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  if (idx < n) {
    array[tid] = smallest[idx];
    array[tid + blockDim.x] = biggest[idx];
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (idx < n && tid % (2*s) == 0) {
      if (tid + s < blockDim.x && idx + s < n) {
        if (array[tid] > array[tid + s]) {
          array[tid] = array[tid + s];
        }
        if (array[tid + blockDim.x] < array[tid + blockDim.x + s]) {
          array[tid + blockDim.x] = array[tid + blockDim.x + s];
        }
      } 
    }
    __syncthreads();
  }
  if (tid == 0) { // Update golbal variables
    smallest[blockIdx.x] = array[0];
    biggest[blockIdx.x] = array[blockDim.x];
  }
}
__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *smallest, int32_t *biggest) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < width * height) {
    idx = (idx % height) * width + idx / height;
    if (smallest[0] != biggest[0]) {
      image[idx] = ((image[idx] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
    }
  }
}
   
 
   
 
 

  



