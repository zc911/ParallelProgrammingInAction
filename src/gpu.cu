#include <cuda.h>
#include <stdio.h>

#include <iostream>

#include "timer.h"

using namespace std;

template <typename T>
struct Mat {
  Mat(int width, int height) : _width(width), _height(height), _data(nullptr) {}
  Mat(int width, int height, T init_value)
      : _width(width), _height(height), _data(nullptr) {
    _data = (T *)malloc(sizeof(T) * _width * _height);
    for (int i = 0; i < _width * _height; i++) _data[i] = init_value;
  }
  ~Mat() {
    if (_data != nullptr) free(_data);
  }
  __host__ __device__ T get(int x, int y) { return _data[y * _width + x]; }
  __device__ void set(int x, int y, T value) { _data[y * _width + x] = value; }

  int _height;
  int _width;
  T *_data;
};

__global__ void blur_mat(Mat<float> *input, Mat<float> *output) {
  int width = input->_width;
  int height = input->_height;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int left = x - 1 < 0 ? 0 : x - 1;
  int right = x + 1 >= width ? width - 1 : x + 1;
  int above = y - 1 < 0 ? 0 : y - 1;
  int below = y + 1 >= height ? height - 1 : y + 1;

  float res = (input->get(x, y) + input->get(left, y) + input->get(right, y) +
               input->get(x, above) + input->get(left, above) +
               input->get(right, above) + input->get(x, below) +
               input->get(left, below) + input->get(right, below)) /
              9;
  output->set(x, y, res);
}

__global__ void blur_mat_redup(Mat<float> *input, Mat<float> *output) {
  int width = input->_width;
  int height = input->_height;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int left = x - 1 < 0 ? 0 : x - 1;
  int right = x + 1 >= width ? width - 1 : x + 1;
  int above = y - 1 < 0 ? 0 : y - 1;
  int below = y + 1 >= height ? height - 1 : y + 1;

  output->set(
      x, y,
      (input->get(x, y) + input->get(left, y) + input->get(right, y)) / 3);
  __syncthreads();
  output->set(
      x, y,
      (output->get(x, y) + output->get(x, above) + output->get(x, below)) / 3);
}

void print_mat(Mat<float> &mat) {
  for (int y = 0; y < mat._height; y++) {
    for (int x = 0; x < mat._width; x++) {
      cout << mat.get(x, y) << ", ";
    }
    cout << endl;
  }
}

int main() {
  cudaSetDevice(3);

  const int width = 8192;
  const int height = 4096;

  Mat<float> *input = new Mat<float>(width, height, 0.0f);
  Mat<float> *output = new Mat<float>(width, height, 0.0f);

  Mat<float> *d_input;
  Mat<float> *d_output;
  Mat<float> *d_input_data = new Mat<float>(width, height);
  Mat<float> *d_output_data = new Mat<float>(width, height);

  cudaMalloc((void **)&d_input, sizeof(Mat<float>));
  cudaMalloc((void **)&d_output, sizeof(Mat<float>));
  cudaMalloc((void **)&(d_input_data->_data), sizeof(float) * width * height);
  cudaMalloc((void **)&(d_output_data->_data), sizeof(float) * width * height);
  cudaMemcpy(d_input, d_input_data, sizeof(Mat<float>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, d_output_data, sizeof(Mat<float>),
             cudaMemcpyHostToDevice);

  Timer t_copy("Host to device");
  cudaMemcpy(d_input_data->_data, input->_data, sizeof(float) * width * height,
             cudaMemcpyHostToDevice);
  t_copy.stop();

  dim3 dim_block(16, 8);
  dim3 dim_grid(width / dim_block.x, height / dim_block.y);

  Timer t1("original");
  blur_mat<<<dim_grid, dim_block>>>(d_input, d_output);
  cudaDeviceSynchronize();
  t1.stop();

  cudaMemcpy(output->_data, d_output_data->_data,
             sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  Timer t2("redup");
  blur_mat_redup<<<dim_grid, dim_block>>>(d_input, d_output);
  cudaDeviceSynchronize();
  t2.stop();

  cudaMemcpy(output->_data, d_output_data->_data,
             sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  printf("%f %f %f\n", output->get(0, 0), output->get(1, 0),
         output->get(100, 100));

  return 0;
}
