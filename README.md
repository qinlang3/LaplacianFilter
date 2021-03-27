# LaplacianFilter
## Introduction:
-  Apply a 5×5 Laplacian filter to images with PGM format (Laplacian of an image is often usedfor edge detection).
-  Analyse and compare the performance of CPU execution and GPU execution for different kernels on this program.
## How to run:
- $ mkdir build\
  $ cd build\
  $ cmake ../\
  $ make
- $ ./pgm_creator.out \<width\> \<height\> \<output_filename\>\
  To create a random PGM image given width and height.
- $ ./main -i \<input_file\> -o \<output_file\>\
  To run the program and printout the performance of CPU execution and GPU execution for different kernels.
