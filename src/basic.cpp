#include <iostream>
#include <vector>

#include "timer.h"

using namespace std;

void blur_mat_original(const vector<vector<int>> &input,
                       vector<vector<int>> &output) {
  int height = input.size();
  int width = input[0].size();
  for (int x = 0; x < width; ++x) {
    int left = x - 1 < 0 ? 0 : x - 1;
    int right = x + 1 >= width ? width - 1 : x + 1;
    for (int y = 0; y < height; ++y) {
      int above = y - 1 < 0 ? 0 : y - 1;
      int below = y + 1 >= height ? height - 1 : y + 1;
      output[y][x] =
          ((input[y][x] + input[y][left] + input[y][right]) +
           (input[above][x] + input[above][left] + input[above][right]) +
           (input[below][x] + input[below][left] + input[below][right])) /
          9;
    }
  }
}

void blur_mat_redup(const vector<vector<int>> &input,
                    vector<vector<int>> &output) {
  int height = input.size();
  int width = input[0].size();
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      int left = x - 1 < 0 ? 0 : x - 1;
      int right = x + 1 >= width ? width - 1 : x + 1;
      output[y][x] = (input[y][x] + input[y][left] + input[y][right]) / 3;
    }
  }

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      int above = y - 1 < 0 ? 0 : y - 1;
      int below = y + 1 >= height ? height - 1 : y + 1;
      output[y][x] = (output[y][x] + output[above][x] + output[below][x]) / 3;
    }
  }
}

void blur_mat_locality(const vector<vector<int>> &input,
                       vector<vector<int>> &output) {
  int height = input.size();
  int width = input[0].size();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int left = x - 1 < 0 ? 0 : x - 1;
      int right = x + 1 >= width ? width - 1 : x + 1;
      output[y][x] = (input[y][x] + input[y][left] + input[y][right]) / 3;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int above = y - 1 < 0 ? 0 : y - 1;
      int below = y + 1 >= height ? height - 1 : y + 1;
      output[y][x] = (output[y][x] + output[above][x] + output[below][x]) / 3;
    }
  }
}

void blur_mat_parallel(const vector<vector<int>> &input,
                       vector<vector<int>> &output) {
  int height = input.size();
  int width = input[0].size();
#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int left = x - 1 < 0 ? 0 : x - 1;
      int right = x + 1 >= width ? width - 1 : x + 1;
      output[y][x] = (input[y][x] + input[y][left] + input[y][right]) / 3;
    }
  }

  // cannot parallel here!
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int above = y - 1 < 0 ? 0 : y - 1;
      int below = y + 1 >= height ? height - 1 : y + 1;
      output[y][x] = (output[y][x] + output[above][x] + output[below][x]) / 3;
    }
  }
}

void blur_mat_tiling(const vector<vector<int>> &input,
                     vector<vector<int>> &output, int tile_width,
                     int tile_height) {
  int height = input.size();
  int width = input[0].size();
  for (int tile_y = 0; tile_y < height / tile_height; ++tile_y) {
    int t_y = tile_y * tile_height;
    for (int tile_x = 0; tile_x < width / tile_width; ++tile_x) {
      int t_x = tile_x * tile_width;
      for (int y = 0; y < tile_height; ++y) {
        int target_y = t_y + y;
        for (int x = 0; x < tile_width; ++x) {
          int target_x = t_x + x;
          int left = target_x - 1 < 0 ? 0 : target_x - 1;
          int right = target_x + 1 >= width ? width - 1 : target_x + 1;
          output[target_y][target_x] =
              (input[target_y][target_x] + input[target_y][left] +
               input[target_y][right]) /
              3;
        }
      }

      for (int y = 0; y < tile_height; ++y) {
        int target_y = t_y + y;
        int above = target_y - 1 < 0 ? 0 : target_y - 1;
        int below = target_y + 1 >= height ? height - 1 : target_y + 1;
        for (int x = 0; x < tile_width; ++x) {
          int target_x = t_x + x;
          output[target_y][target_x] =
              (output[target_y][target_x] + output[target_y][above] +
               output[target_y][below]) /
              3;
        }
      }
    }
  }
}

void blur_mat_tiling_parallel(const vector<vector<int>> &input,
                              vector<vector<int>> &output, int tile_width,
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
          int left = target_x - 1 < 0 ? 0 : target_x - 1;
          int right = target_x + 1 >= width ? width - 1 : target_x + 1;
          output[target_y][target_x] =
              (input[target_y][target_x] + input[target_y][left] +
               input[target_y][right]) /
              3;
        }
      }

      for (int y = 0; y < tile_height; ++y) {
        int target_y = t_y + y;
        int above = target_y - 1 < 0 ? 0 : target_y - 1;
        int below = target_y + 1 >= height ? height - 1 : target_y + 1;
        for (int x = 0; x < tile_width; ++x) {
          int target_x = t_x + x;
          output[target_y][target_x] =
              (output[target_y][target_x] + output[target_y][above] +
               output[target_y][below]) /
              3;
        }
      }
    }
  }
}

void print_mat(const vector<vector<int>> &mat) {
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

  vector<vector<int>> in_data(height, vector<int>(width, 1));

  vector<vector<int>> out_data_1(height, vector<int>(width, 0));
  Timer t1("1 original");
  blur_mat_original(in_data, out_data_1);
  t1.stop();

  vector<vector<int>> out_data_2(height, vector<int>(width, 0));
  Timer t2("2 redup");
  blur_mat_redup(in_data, out_data_2);
  t2.stop();

  vector<vector<int>> out_data_3(height, vector<int>(width, 0));
  Timer t3("3 locality");
  blur_mat_locality(in_data, out_data_3);
  t3.stop();

  vector<vector<int>> out_data_5(height, vector<int>(width, 0));
  Timer t5("5 parallel");
  blur_mat_parallel(in_data, out_data_5);
  t5.stop();

  vector<vector<int>> out_data_4_6(height, vector<int>(width, 0));
  Timer t4_6("6 1024*512 titing");
  blur_mat_tiling(in_data, out_data_4_6, 1024, 512);
  t4_6.stop();

  vector<vector<int>> out_data_7(height, vector<int>(width, 0));
  Timer t7("7 tiling + parallel");
  blur_mat_tiling_parallel(in_data, out_data_7, 1024, 256);
  t7.stop();

  // vector<vector<int>> out_data_4_1(height, vector<int>(width, 0));
  // Timer t4_1("4_1 2*2 titing");
  // blur_mat_tiling(in_data, out_data_4_1, 2, 2);
  // t4_1.stop();

  // vector<vector<int>> out_data_4_2(height, vector<int>(width, 0));
  // Timer t4_2("4_2 8*8 titing");
  // blur_mat_tiling(in_data, out_data_4_2, 8, 8);
  // t4_2.stop();

  // vector<vector<int>> out_data_4_3(height, vector<int>(width, 0));
  // Timer t4_3("4_3 32*32  titing");
  // blur_mat_tiling(in_data, out_data_4_3, 32, 32);
  // t4_3.stop();

  // vector<vector<int>> out_data_4_4(height, vector<int>(width, 0));
  // Timer t4_4("4_4 64*64 titing");
  // blur_mat_tiling(in_data, out_data_4_4, 64, 64);
  // t4_4.stop();

  // vector<vector<int>> out_data_4_5(height, vector<int>(width, 0));
  // Timer t4_5("4_5 256*256 titing");
  // blur_mat_tiling(in_data, out_data_4_5, 256, 256);
  // t4_5.stop();

  return 0;
}