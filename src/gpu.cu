#include <cuda.h>
#include <stdio.h>

#include <iostream>

using namespace std;

__global__ void blur_mat(float **input, float **output) {
  // cout << "hello CUDA" << endl;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  output[y][x] = input[y][x] + 1;
  //   printf("%f", input[y][x]);
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
  const int width = 8;
  const int height = 4;

  printf("1\n");

  float **input = (float **)malloc(sizeof(float *) * height);
  float **output = (float **)malloc(sizeof(float *) * height);
  float *input_data = (float *)malloc(sizeof(float) * width * height);
  float *output_data = (float *)malloc(sizeof(float) * width * height);
  printf("2\n");
  for (int i = 0; i < width * height; ++i) {
    input_data[i] = 1.0f;
    output_data[i] = 0.0f;
  }
  printf("3\n");
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
    output[i] = d_output_data + height * i;
  }
  printf("4\n");
  cudaMemcpy(d_input, input, sizeof(float *) * height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, sizeof(float *) * height,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_data, input_data, sizeof(float) * width * height,
             cudaMemcpyHostToDevice);

  dim3 dim_block(4, 4);
  dim3 dim_grid(width / dim_block.x, height / dim_block.y);
  printf("5\n");
  blur_mat<<<dim_grid, dim_block>>>(d_input, d_output);
  printf("6\n");
  cudaMemcpy(output_data, d_output_data, sizeof(float) * width * height,
             cudaMemcpyDeviceToHost);

  print_mat(output_data, width, height);

  cudaDeviceReset();
  return 0;
}
