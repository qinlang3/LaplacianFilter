#include <stdio.h>
#include <string>
#include <unistd.h>
 
#include "pgm.h"
#include "clock.h"
#include "kernels.h"
 
// The 5 * 5 Laplacian filter
const int8_t f[] = {
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
 
/* Use this function to print the time of each of your kernels.
* The parameter names are intuitive, but don't hesitate to ask
* for clarifications.
* DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
                float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                 time_gpu_transfer_out));
}
 
int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }
 
  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
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
 
  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }
 
  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
          "Speedup_noTrf Speedup\n");
 
  /* TODO: run your CPU implementation here and get its time. Don't include
  * file IO in your measurement.*/
  /* For example: */
  float time_cpu;
  {
    std::string cpu_file = cpu_output_filename;
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);
    // Start time
    Clock clock;
    clock.start();
    run_best_cpu(f, 5, source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height);  // From kernels.h
    // End time
    time_cpu = clock.stop();
    // print_run(args...)      // Defined on the top of this file
    save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
    destroy_pgm_image(&cpu_output_img);
  }
 
  {
    std::string gpu_file = "1" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    int8_t *d_filter = NULL;
    size_t size = source_img.width * source_img.height * sizeof(int32_t);
    cudaMalloc((void **)&d_filter, 25 * sizeof(int8_t));
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    float transfer_in, transfer_out, computation;
 
    Clock clock;
    clock.start();
    cudaMemcpy(d_filter, f, 25 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
    transfer_in = clock.stop();
 
    // Start time
    clock.start();
    run_kernel1(d_filter, 5, d_input, d_output, gpu_output_img.width, gpu_output_img.height);
    // End time
    computation = clock.stop();
    
    clock.start();
    cudaMemcpy(gpu_output_img.matrix, d_output, size, cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();
 
    print_run(time_cpu, 1, computation, transfer_in, transfer_out); 
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    destroy_pgm_image(&gpu_output_img);
  }
   
  {
    std::string gpu_file = "2" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    int8_t *d_filter = NULL;
    size_t size = source_img.width * source_img.height * sizeof(int32_t);
    cudaMalloc((void **)&d_filter, 25 * sizeof(int8_t));
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    float transfer_in, transfer_out, computation;
 
    Clock clock;
    clock.start();
    cudaMemcpy(d_filter, f, 25 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
    transfer_in = clock.stop();
 
    // Start time
    clock.start();
    run_kernel2(d_filter, 5, d_input, d_output, gpu_output_img.width, gpu_output_img.height);
    // End time
    computation = clock.stop();
    
    clock.start();
    cudaMemcpy(gpu_output_img.matrix, d_output, size, cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();
 
    print_run(time_cpu, 2, computation, transfer_in, transfer_out); 
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    destroy_pgm_image(&gpu_output_img);
  }

  {
    std::string gpu_file = "3" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    int8_t *d_filter = NULL;
    size_t size = source_img.width * source_img.height * sizeof(int32_t);
    cudaMalloc((void **)&d_filter, 25 * sizeof(int8_t));
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    float transfer_in, transfer_out, computation;

    Clock k;
    k.start();
    cudaMemcpy(d_filter, f, 25 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
    transfer_in = k.stop();

    // Start time
    k.start();
    run_kernel3(d_filter, 5, d_input, d_output, gpu_output_img.width, gpu_output_img.height);
    // End time
    computation = k.stop();
    k.start();
    cudaMemcpy(gpu_output_img.matrix, d_output, size, cudaMemcpyDeviceToHost);
    transfer_out = k.stop();

    print_run(time_cpu, 3, computation, transfer_in, transfer_out); 
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    destroy_pgm_image(&gpu_output_img);
  }

  {
    std::string gpu_file = "4" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    int8_t *d_filter = NULL;
    size_t size = source_img.width * source_img.height * sizeof(int32_t);
    cudaMalloc((void **)&d_filter, 25 * sizeof(int8_t));
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    float transfer_in, transfer_out, computation;

    Clock k;
    k.start();
    cudaMemcpy(d_filter, f, 25 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
    transfer_in = k.stop();

    // Start time
    k.start();
    run_kernel4(d_filter, 5, d_input, d_output, gpu_output_img.width, gpu_output_img.height);
    // End time
    computation = k.stop();
    k.start();
    cudaMemcpy(gpu_output_img.matrix, d_output, size, cudaMemcpyDeviceToHost);
    transfer_out = k.stop();

    print_run(time_cpu, 4, computation, transfer_in, transfer_out); 
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    destroy_pgm_image(&gpu_output_img);
  }

  {
    std::string gpu_file = "5" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    int8_t *d_filter = NULL;
    size_t size = source_img.width * source_img.height * sizeof(int32_t);
    cudaMalloc((void **)&d_filter, 25 * sizeof(int8_t));
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    float transfer_in, transfer_out, computation;
 
    Clock clock;
    clock.start();
    cudaMemcpy(d_filter, f, 25 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
    transfer_in = clock.stop();
 
    // Start time
    clock.start();
    run_kernel5(d_filter, 5, d_input, d_output, gpu_output_img.width, gpu_output_img.height);
     
    // End time
    computation = clock.stop();
    
    clock.start();
    cudaMemcpy(gpu_output_img.matrix, d_output, size, cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();
 
    print_run(time_cpu, 5, computation, transfer_in, transfer_out); 
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    destroy_pgm_image(&gpu_output_img);
  }  
}
 