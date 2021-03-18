#include <pmmintrin.h>
// #include <xmmintrin.h>

#include <iostream>
#include <vector>

#include "timer.h"

using namespace std;

void blur_mat_original(const vector<vector<float>> &input,
                       vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
  int right, right_right, below, below_below;
  for (int x = 0; x < width; ++x) {
    right = x + 1 >= width ? width - 1 : x + 1;
    right_right = x + 2 >= width ? width - 1 : x + 2;
    for (int y = 0; y < height; ++y) {
      below = y + 1 >= height ? height - 1 : y + 1;
      below_below = y + 2 >= height ? height - 1 : y + 2;
      output[y][x] =
          ((input[y][x] + input[y][right] + input[y][right_right]) +
           (input[below][x] + input[below][right] + input[below][right_right]) +
           (input[below_below][x] + input[below_below][right] +
            input[below_below][right_right])) /
          9;
    }
  }
}

void blur_mat_redup(const vector<vector<float>> &input,
                    vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
  int right, right_right, below, below_below;
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      right = x + 1 >= width ? width - 1 : x + 1;
      right_right = x + 2 >= width ? width - 1 : x + 2;
      output[y][x] =
          (input[y][x] + input[y][right] + input[y][right_right]) / 3;
    }
  }
  // not equivalence
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      below = y + 1 >= height ? height - 1 : y + 1;
      below_below = y + 2 >= height ? height - 1 : y + 2;
      output[y][x] =
          (output[y][x] + output[below][x] + output[below_below][x]) / 3;
    }
  }
}

void blur_mat_locality(const vector<vector<float>> &input,
                       vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
  int right, right_right, below, below_below;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      right = x + 1 >= width ? width - 1 : x + 1;
      right_right = x + 2 >= width ? width - 1 : x + 2;
      output[y][x] =
          (input[y][x] + input[y][right] + input[y][right_right]) / 3;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      below = y + 1 >= height ? height - 1 : y + 1;
      below_below = y + 2 >= height ? height - 1 : y + 2;
      output[y][x] =
          (output[y][x] + output[below][x] + output[below_below][x]) / 3;
    }
  }
}

void blur_mat_parallel(const vector<vector<float>> &input,
                       vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
#pragma omp parallel for
  for (int x = 0; x < width; ++x) {
    int right = x + 1 >= width ? width - 1 : x + 1;
    int right_right = x + 2 >= width ? width - 1 : x + 2;
    for (int y = 0; y < height; ++y) {
      int below = y + 1 >= height ? height - 1 : y + 1;
      int below_below = y + 2 >= height ? height - 1 : y + 2;
      output[y][x] =
          ((input[y][x] + input[y][right] + input[y][right_right]) +
           (input[below][x] + input[below][right] + input[below][right_right]) +
           (input[below_below][x] + input[below_below][right] +
            input[below_below][right_right])) /
          9;
    }
  }
}

void blur_mat_parallel_redup(const vector<vector<float>> &input,
                             vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int right = x + 1 >= width ? width - 1 : x + 1;
      int right_right = x + 2 >= width ? width - 1 : x + 2;
      output[y][x] =
          (input[y][x] + input[y][right] + input[y][right_right]) / 3;
    }
  }
  // can not parallel here
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int below = y + 1 >= height ? height - 1 : y + 1;
      int below_below = y + 2 >= height ? height - 1 : y + 2;
      output[y][x] =
          (output[y][x] + output[below][x] + output[below_below][x]) / 3;
    }
  }
}

void blur_mat_tiling(const vector<vector<float>> &input,
                     vector<vector<float>> &output, int tile_width,
                     int tile_height) {
  int height = input.size();
  int width = input[0].size();
  int t_y, t_x, target_y, target_x, right, right_right, below, below_below;
  for (int tile_y = 0; tile_y < height / tile_height; ++tile_y) {
    t_y = tile_y * tile_height;
    for (int tile_x = 0; tile_x < width / tile_width; ++tile_x) {
      t_x = tile_x * tile_width;
      for (int y = 0; y < tile_height; ++y) {
        target_y = t_y + y;
        for (int x = 0; x < tile_width; ++x) {
          target_x = t_x + x;
          right = target_x + 1 >= width ? width - 1 : target_x + 1;
          right_right = target_x + 2 >= width ? width - 1 : target_x + 2;
          output[target_y][target_x] =
              (input[target_y][target_x] + input[target_y][right] +
               input[target_y][right_right]) /
              3;
        }
      }

      for (int y = 0; y < tile_height; ++y) {
        target_y = t_y + y;
        below = target_y + 1 >= height ? height - 1 : target_y + 1;
        below_below = target_y + 1 >= height ? height - 1 : target_y + 1;
        for (int x = 0; x < tile_width; ++x) {
          target_x = t_x + x;
          output[target_y][target_x] =
              (output[target_y][target_x] + output[below][target_x] +
               output[below_below][target_x]) /
              3;
        }
      }
    }
  }
}

void blur_mat_tiling_parallel(const vector<vector<float>> &input,
                              vector<vector<float>> &output, int tile_width,
                              int tile_height) {
  int height = input.size();
  int width = input[0].size();
#pragma omp parallel for
  for (int tile_y = 0; tile_y < height / tile_height; ++tile_y) {
    int t_y = tile_y * tile_height;
    for (int tile_x = 0; tile_x < width / tile_width; ++tile_x) {
      int t_x = tile_x * tile_width;
      for (int y = 0; y < tile_height; ++y) {
        int target_y = t_y + y;
        for (int x = 0; x < tile_width; ++x) {
          int target_x = t_x + x;
          int right = target_x + 1 >= width ? width - 1 : target_x + 1;
          int right_right = target_x + 2 >= width ? width - 1 : target_x + 2;
          output[target_y][target_x] =
              (input[target_y][target_x] + input[target_y][right] +
               input[target_y][right_right]) /
              3;
        }
      }

      for (int y = 0; y < tile_height; ++y) {
        int target_y = t_y + y;
        int below = target_y + 1 >= height ? height - 1 : target_y + 1;
        int below_below = target_y + 2 >= height ? height - 1 : target_y + 2;
        for (int x = 0; x < tile_width; ++x) {
          int target_x = t_x + x;
          output[target_y][target_x] =
              (output[target_y][target_x] + output[below][target_x] +
               output[below_below][target_x]) /
              3;
        }
      }
    }
  }
}

void blur_mat_sse(const vector<vector<float>> &input,
                  vector<vector<float>> &output) {
  int height = input.size();
  int width = input[0].size();
#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int below = y + 1 >= height ? height - 1 : y + 1;
      int below_below = y + 2 >= height ? height - 1 : y + 2;
      __m128 vdata_1 = _mm_loadu_ps(&input[y][x]);
      __m128 vdata_2 = _mm_loadu_ps(&input[below][x]);
      __m128 vdata_3 = _mm_loadu_ps(&input[below_below][x]);
      __m128 vres = _mm_add_ps(vdata_1, vdata_2);
      vres = _mm_add_ps(vres, vdata_3);
      vres = _mm_hadd_ps(vres, vres);
      vres = _mm_hadd_ps(vres, vres);
      _mm_store_ss(&output[y][x], vres);
      output[y][x] /= 12;
    }
  }
}

void print_mat(const vector<vector<float>> &mat) {
  for (auto row : mat) {
    for (auto ele : row) {
      cout << ele << ", ";
    }
    cout << endl;
  }
}

int main() {
  const int width = 8192;
  const int height = 4096;

  vector<vector<float>> in_data(height, vector<float>(width, 1));

  vector<vector<float>> out_data_1(height, vector<float>(width, 0));
  Timer t1("1 original");
  blur_mat_original(in_data, out_data_1);
  t1.stop();

  vector<vector<float>> out_data_2(height, vector<float>(width, 0));
  Timer t2("2 redup", t1.get());
  blur_mat_redup(in_data, out_data_2);
  t2.stop();

  vector<vector<float>> out_data_3(height, vector<float>(width, 0));
  Timer t3("3 locality", t1.get());
  blur_mat_locality(in_data, out_data_3);
  t3.stop();

  vector<vector<float>> out_data_4(height, vector<float>(width, 0));
  Timer t4("4 parallel", t1.get());
  blur_mat_parallel(in_data, out_data_4);
  t4.stop();

  vector<vector<float>> out_data_5(height, vector<float>(width, 0));
  Timer t5("5 parallel + redup", t1.get());
  blur_mat_parallel_redup(in_data, out_data_5);
  t5.stop();

  vector<vector<float>> out_data_4_6(height, vector<float>(width, 0));
  Timer t4_6("6 1024*512 titing", t1.get());
  blur_mat_tiling(in_data, out_data_4_6, 1024, 512);
  t4_6.stop();

  vector<vector<float>> out_data_7(height, vector<float>(width, 0));
  Timer t7("7 tiling + parallel", t1.get());
  blur_mat_tiling_parallel(in_data, out_data_7, 1024, 256);
  t7.stop();

  vector<vector<float>> out_data_8(height, vector<float>(width, 0));
  Timer t8("8 parallel + sse", t1.get());
  blur_mat_sse(in_data, out_data_8);
  t8.stop();

  return 0;
}