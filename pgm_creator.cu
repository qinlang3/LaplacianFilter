#include "pgm.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Wrong usage: ./pgm_creator.out <width> <height>"
           "<output_filename>\n");
    return 0;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);

  pgm_image image;

  create_random_pgm_image(&image, width, height);
  int32_t err = save_pgm_to_file(argv[3], &image);

  if (err != NO_ERR) {
    printf("ERR = %d\n", err);
    return 1;
  }

  return 0;
}
