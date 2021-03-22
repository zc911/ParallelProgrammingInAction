#include <cuda.h>
#include <stdio.h>

#include <iostream>

#include "timer.h"

using namespace std;

__global__ void blur_mat(float **input, float **output, int width, int height) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int left = x - 1 < 0 ? 0 : x - 1;
  int right = x + 1 >= width ? width - 1 : x + 1;
  int above = y - 1 < 0 ? 0 : y - 1;
  int below = y + 1 >= height ? height - 1 : y + 1;
  output[y][x] = (input[y][x] + input[y][left] + input[y][right] +
                  input[above][left] + input[above][x] + input[above][right] +
                  input[below][left] + input[below][x] + input[below][right]) /
                 9;
}

__global__ void blur_mat_redup(float **input, float **output, int width,
                               int height) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int left = x - 1 < 0 ? 0 : x - 1;
  int right = x + 1 >= width ? width - 1 : x + 1;
  int above = y - 1 < 0 ? 0 : y - 1;
  int below = y + 1 >= height ? height - 1 : y + 1;
  output[y][x] = (input[y][x] + input[y][left] + input[y][right]) / 3;
  output[y][x] = (output[y][x] + output[above][x] + output[below][x]) / 3;
}

void print_mat(float *data, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cout << data[height * y + x] << ", ";
    }
    cout << endl;
  }
}

int main() {
  cudaSetDevice(3);
  const int width = 8192;
  const int height = 4096;

  float **input = (float **)malloc(sizeof(float *) * height);
  float **output = (float **)malloc(sizeof(float *) * height);
  float *input_data = (float *)malloc(sizeof(float) * width * height);
  float *output_data = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; ++i) {
    input_data[i] = i;
    output_data[i] = 0.0f;
  }

  float **d_input;
  float **d_output;
  float *d_input_data;
  float *d_output_data;

  cudaMalloc((void **)&d_input, sizeof(float **) * height);
  cudaMalloc((void **)&d_output, sizeof(float **) * height);
  cudaMalloc((void **)&d_input_data, sizeof(float) * width * height);
  cudaMalloc((void **)&d_output_data, sizeof(float) * width * height);

  for (int i = 0; i < height; ++i) {
    input[i] = d_input_data + width * i;
    output[i] = d_output_data + width * i;
  }

  Timer t_copy("Host to device");
  cudaMemcpy(d_input, input, sizeof(float *) * height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, sizeof(float *) * height,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_data, input_data, sizeof(float) * width * height,
             cudaMemcpyHostToDevice);
  t_copy.stop();

  dim3 dim_block(16, 8);
  dim3 dim_grid(width / dim_block.x, height / dim_block.y);

  Timer t1("original");
  blur_mat<<<dim_grid, dim_block>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
  t1.stop();

  cudaMemcpy(output_data, d_output_data, sizeof(float) * width * height,
             cudaMemcpyDeviceToHost);

  Timer t2("redup");
  blur_mat_redup<<<dim_grid, dim_block>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
  t2.stop();

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_input_data);
  cudaFree(d_output_data);

  printf("%f,%f\n", output_data[0], output_data[1200]);

  free(input);
  free(output);
  free(input_data);
  free(output_data);

  return 0;
}
