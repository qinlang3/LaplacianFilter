#include "kernels.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sched.h>

/* Gobal variables */
int32_t *min_array;
int32_t *max_array;
cpu_set_t set;

/* Data structure to store shared data for sharding_work. */
typedef struct common_work_t
{
    const int8_t *f;
    int32_t dimension;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;
} common_work;

/* Data structure to store arguments passed into each thread
 * for sharding_work. */
typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;


/* Calcualte the line difference between two pixels
 * in a matrix given its width. */


void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, int32_t largest) {
    if (smallest == largest) {
        return;
    }
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *original, int32_t *target, 
                    int32_t width, int32_t height, int row, int column) {
    int32_t i = row * width + column;
    if (dimension == 1) {
        return original[i] * f[0];
    }else{
        int32_t j;
        int32_t acc = 0;
        int mid, line, k;
        int counter = 0;
        if (dimension == 3){
          mid = 1;
          line = 3;
        }
        if (dimension == 5) {
          mid = 2;
          line = 5;
        }
        if (dimension == 9) {
          mid = 4;
          line = 9;
        }
        for (k = 0; k < line; k++) {
            for (j = i - (mid-k)*width - mid; j < i - (mid-k)*width + mid + 1; j++) {
                if ((j >= 0 && j < width*height) && ((i/width)-(j/width)+k == mid)) {
                    acc += f[counter] * original[j]; 
                }
                counter++;
            }
        }
        return acc;
    }  
}


void *sharding_work(void *work) {
    work_t *data;
    data = (work_t *) work;
    const int8_t *f; 
    int32_t dimension;
    const int32_t *original;
    int32_t *target;
    int32_t id, width, height, max_threads, assigned_work, minimum, maximum, result, working_threads, i, j;
    id = data->id;
    
    CPU_SET(id, &set);
    if (sched_setaffinity(getpid(), sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
    }
    f = data->common->f;
    dimension = data->common->dimension;
    original = data->common->original_image;
    target = data->common->output_image;
    width = data->common->width;
    height = data->common->height;
    max_threads = data->common->max_threads;
    
    // Calculate amount of work assgined to each thread and the 
    // number of actul working threads based on different ways of sharding, 
    // stored in assigned_work and working_threads respectively.
    if (height % max_threads > 0) {
        assigned_work = (height + max_threads - 1) / max_threads;
    }else{
        assigned_work = height / max_threads;
    }
    if (height % assigned_work > 0) {
        working_threads = height / assigned_work + 1;
    }else{
        working_threads = height / assigned_work;
    }
    
    // Image processing starts.
    int first = 1;
    if (id < working_threads) {
      // Image processing for SHARDED_ROWS.
        for (i = 0; i < assigned_work; i++) {
            if (id * assigned_work + i < height) {
                for (j = 0; j < width; j++) {
                    result = apply2d(f, dimension, original, target, width, height, id * assigned_work + i, j);
                    target[(id * assigned_work + i) * width + j] = result;
                    if (first) {
                        minimum = result;
                        maximum = result;
                        first = 0;
                    }else{
                        if (result < minimum) {
                            minimum = result;
                        }
                        if (result > maximum) {
                            maximum = result;
                        }
                    }
                }
            }
        }
        min_array[id] = minimum;
        max_array[id] = maximum;
    }
    // Threads waiting for each other to finish finding local 
    // Min and loacl Max.
    pthread_barrier_wait(&data->common->barrier);
  
    // Calculate the global Min and global Max.
    if (id < working_threads) {
        minimum = min_array[0];
        maximum = max_array[0];
        for (i = 0; i < working_threads; i++) {
            if (min_array[i] < minimum){
                minimum = min_array[i];
            }
            if (max_array[i] > maximum){
                maximum = max_array[i];
            }
        }

        for (i = 0; i < assigned_work; i++) {
            if (id * assigned_work + i < height) {
                for (j = 0; j < width; j++) {
                    normalize_pixel(target, (id * assigned_work + i) * width + j, minimum, maximum);
                }
            }  
        }
    }
    pthread_exit(0);
  }
   

   

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
    int i;
    // Using 8 threads.
    int num_threads = 8;
    pthread_t thread_array[num_threads];
    pthread_barrier_t barrier;
    CPU_ZERO(&set);
                    
    common_work_t common;
    common.f = filter;
    common.dimension = dimension;
    common.original_image = input;
    common.output_image = output;
    common.width = width;
    common.height = height;
    common.max_threads = num_threads;
    pthread_barrier_init(&barrier, NULL, num_threads);
    common.barrier = barrier;
                     
    work_t work_array[num_threads];
    min_array = (int32_t *) malloc(sizeof(int32_t) * num_threads);
    max_array = (int32_t *) malloc(sizeof(int32_t) * num_threads);
    for (i = 0; i < num_threads; i++) {
        work_array[i].common = &common;
        work_array[i].id = i;
        pthread_create(&thread_array[i], NULL, sharding_work, &work_array[i]);
    }
    for (i = 0; i < num_threads; i++) {
        pthread_join(thread_array[i], NULL);
    }  
    free(min_array);
    free(max_array);
}
