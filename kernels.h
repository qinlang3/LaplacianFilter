#ifndef __KERNELS__H
#define __KERNELS__H

/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions
 * unfortunately, so don't use those for variable names.*/

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void reduction(int32_t *smallest, int32_t *biggest, int n);
__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void reduction4(int32_t *gmin, int32_t *gmax, int n);
__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
/* This is your own kernel, you should decide which parameters to add
   here*/

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *gmin, int32_t *gmax);
__global__ void reduction5(int32_t *smallest, int32_t *biggest, int n);
__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

#endif
