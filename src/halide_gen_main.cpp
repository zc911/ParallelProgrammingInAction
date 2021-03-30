#include <stdio.h>

#include "Halide.h"
#include "auto_blur_mat.h"
#include "blur_mat.h"
#include "timer.h"

using namespace Halide;

#define WIDTH 8192
#define HEIGHT 4096

int main() {
  Halide::Runtime::Buffer<float> input(WIDTH, HEIGHT);
  Halide::Runtime::Buffer<float> output(WIDTH, HEIGHT);
  for (int x = 0; x < WIDTH; ++x) {
    for (int y = 0; y < HEIGHT; ++y) {
      input(x, y) = x * WIDTH + y;
    }
  }

  Timer t1("AOT");
  blur_mat(input, output);
  t1.stop();

  Timer t2("Auto");
  auto_blur_mat(input, output);
  t2.stop();

  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      printf("%f,", output(x, y));
    }
    printf("\n");
  }
  return 0;
}