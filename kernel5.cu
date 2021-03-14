#include "kernels.h"
#define threads_block 512
#define thread_pixel 16
#include <stdio.h>
 
void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
  // Calculate threads and blocks.
  int num_threads, num_blocks;
  num_threads = (width * height + thread_pixel - 1) / thread_pixel;
  if (num_threads > threads_block) {
    num_blocks = (num_threads + threads_block - 1) / threads_block;
    num_threads = threads_block;
  }else {
    num_blocks = 1;
  }
  // Initialiaze golbal device variables to store minimum and maximum.
  int32_t *gmin = NULL;
  int32_t *gmax = NULL;
  size_t size = num_blocks * sizeof(int32_t);
  cudaMalloc((void **)&gmin, size);
  cudaMalloc((void **)&gmax, size);
  cudaMemcpy(gmin, output, size, cudaMemcpyDeviceToDevice);
  cudaMemcpy(gmax, output, size, cudaMemcpyDeviceToDevice);

  dim3 dimBlock(num_threads, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);
  int shMemSize = (num_threads <= 32) ? 4 * num_threads * sizeof(int32_t) : 2* num_threads * sizeof(int32_t);
  kernel5<<<dimGrid, dimBlock, shMemSize>>>(filter, dimension, input, output, width, height, gmin, gmax);

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
    reduction5<<<newgrid, newblock, shMemSize>>>(gmin, gmax, n);
  }
  // Calling normalize kernel.
  normalize5<<<dimGrid, dimBlock>>>(output, width, height, gmin, gmax);
  cudaFree(gmin);
  cudaFree(gmax);
}
 
__global__ void kernel5(const int8_t *filter, int32_t dimension,
                         const int32_t *input, int32_t *output, int32_t width,
                         int32_t height, int32_t *gmin, int32_t *gmax) 
{
  extern __shared__ int32_t array[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockSize = blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int32_t minimum, maximum;
  int first = 1;
  for (int i = idx; i < width * height; i+=stride) {
    int32_t j;
    int counter = 0;
    int32_t acc = 0;
    for (int k = 0; k < 5; k++) {
      for (j = i - (2-k)*width - 2; j < i - (2-k)*width + 3; j++) {
        if ((j >= 0 && j < width*height) && ((i/width)-(j/width)+k == 2)) {
          if (counter != 12) {
            acc += -1 * input[j];
          }else {
            acc += 24 * input[j];
          }
        }
        counter++;
      }
    }
    output[i] = acc;
    if (first) {
      minimum = acc;
      maximum = acc;
      first = 0;
    }else {
      if (minimum > acc) { minimum = acc;}
      if (maximum < acc) { maximum = acc;}
    }
  }

  // Reduction step
  if (idx < width * height) {
    array[tid] = minimum;
    array[tid + blockSize] = maximum;
  }
  __syncthreads();
  
  if (blockSize >= 512) { if (idx + 256 < width*height && tid < 256 && tid+256 < blockSize){ if (array[tid] > array[tid + 256]) { array[tid] = array[tid + 256];} 
              if (array[tid + blockSize] < array[tid + blockSize + 256]) { array[tid + blockSize] = array[tid + blockSize + 256];}} __syncthreads();}
  if (blockSize >= 256) { if (idx + 128 < width*height && tid < 128 && tid+128 < blockSize){ if (array[tid] > array[tid + 128]) { array[tid] = array[tid + 128];} 
              if (array[tid + blockSize] < array[tid + blockSize + 128]) { array[tid + blockSize] = array[tid + blockSize + 128];}} __syncthreads();}
  if (blockSize >= 128) { if (idx + 64 < width*height && tid < 64 && tid+64 < blockSize){ if (array[tid] > array[tid + 64]) { array[tid] = array[tid + 64];} 
              if (array[tid + blockSize] < array[tid + blockSize + 64]) { array[tid + blockSize] = array[tid + blockSize + 64];}} __syncthreads();}
	
  if (tid < 32) {
		volatile int* smem = array;
		if (blockSize >= 64 && tid + 32 < blockSize && idx + 32 < width*height) {
      smem[tid] += smem[tid + 32];
      smem[tid + blockSize] = smem[tid + blockSize + 32];
    }
    if (blockSize >= 16 && tid + 16 < blockSize && idx + 16 < width*height) {
      smem[tid] += smem[tid + 16];
      smem[tid + blockSize] = smem[tid + blockSize + 16];
    }
    if (blockSize >= 64 && tid + 8 < blockSize && idx + 8 < width*height) {
      smem[tid] += smem[tid + 8];
      smem[tid + blockSize] = smem[tid + blockSize + 8];
    }
    if (blockSize >= 64 && tid + 4 < blockSize && idx + 4 < width*height) {
      smem[tid] += smem[tid + 4];
      smem[tid + blockSize] = smem[tid + blockSize + 4];
    }
    if (blockSize >= 64 && tid + 2 < blockSize && idx + 2 < width*height) {
      smem[tid] += smem[tid + 2];
      smem[tid + blockSize] = smem[tid + blockSize + 2];
    }
    if (blockSize >= 64 && tid + 1 < blockSize && idx + 1 < width*height) {
      smem[tid] += smem[tid + 1];
      smem[tid + blockSize] = smem[tid + blockSize + 1];
    } 
	} 
      
  if (tid == 0) { // Update golbal variables
    gmin[blockIdx.x] = array[0];
    gmax[blockIdx.x] = array[blockSize];
  }  
}

__global__ void reduction5(int32_t *smallest, int32_t *biggest, int n) {
  extern __shared__ int32_t array[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockSize = blockDim.x;
  if (idx < n) {
    array[tid] = smallest[idx];
    array[tid + blockSize] = biggest[idx];
  }
  __syncthreads();
  
  if (blockSize >= 512) { if (idx + 256 < n && tid < 256 && tid+256 < blockSize){ if (array[tid] > array[tid + 256]) { array[tid] = array[tid + 256];} 
              if (array[tid + blockSize] < array[tid + blockSize + 256]) { array[tid + blockSize] = array[tid + blockSize + 256];}} __syncthreads();}
  if (blockSize >= 256) { if (idx + 128 < n && tid < 128 && tid+128 < blockSize){ if (array[tid] > array[tid + 128]) { array[tid] = array[tid + 128];} 
              if (array[tid + blockSize] < array[tid + blockSize + 128]) { array[tid + blockSize] = array[tid + blockSize + 128];}} __syncthreads();}
  if (blockSize >= 128) { if (idx + 64 < n && tid < 64 && tid+64 < blockSize){ if (array[tid] > array[tid + 64]) { array[tid] = array[tid + 64];} 
              if (array[tid + blockSize] < array[tid + blockSize + 64]) { array[tid + blockSize] = array[tid + blockSize + 64];}} __syncthreads();}
	
  if (tid < 32) {
		volatile int* smem = array;
		if (blockSize >= 64 && tid + 32 < blockSize && idx + 32 < n) {
      smem[tid] += smem[tid + 32];
      smem[tid + blockSize] = smem[tid + blockSize + 32];
    }
    if (blockSize >= 16 && tid + 16 < blockSize && idx + 16 < n) {
      smem[tid] += smem[tid + 16];
      smem[tid + blockSize] = smem[tid + blockSize + 16];
    }
    if (blockSize >= 64 && tid + 8 < blockSize && idx + 8 < n) {
      smem[tid] += smem[tid + 8];
      smem[tid + blockSize] = smem[tid + blockSize + 8];
    }
    if (blockSize >= 64 && tid + 4 < blockSize && idx + 4 < n) {
      smem[tid] += smem[tid + 4];
      smem[tid + blockSize] = smem[tid + blockSize + 4];
    }
    if (blockSize >= 64 && tid + 2 < blockSize && idx + 2 < n) {
      smem[tid] += smem[tid + 2];
      smem[tid + blockSize] = smem[tid + blockSize + 2];
    }
    if (blockSize >= 64 && tid + 1 < blockSize && idx + 1 < n) {
      smem[tid] += smem[tid + 1];
      smem[tid + blockSize] = smem[tid + blockSize + 1];
    } 
	} 
      
  if (tid == 0) { // Update golbal variables
    smallest[blockIdx.x] = array[0];
    biggest[blockIdx.x] = array[blockSize];
  }  
 
}
 __global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                            int32_t *smallest, int32_t *biggest) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < width * height; i+=stride) {
    if (smallest[0] != biggest[0]) {
      image[i] = ((image[i] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
    }
  }
}
 
