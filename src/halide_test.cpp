#include "Halide.h"

#include <stdio.h>

#include <iostream>

#include "timer.h"

using namespace std;
using namespace Halide;

#define WIDTH 8192
#define HEIGHT 4096

void blur_original(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;
  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_x_root(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

  blur_x.compute_root();
  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_x_at_y(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

  blur_x.compute_at(blur_y, y);
  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_x_store(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;
  blur_x.store_root();
  blur_x.compute_at(blur_y, y);
  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_x_at_x(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

  blur_x.compute_at(blur_y, x);
  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_tile(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;

  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

  Var x_outer, y_outer, x_inner, y_inner;
  blur_y.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
      .parallel(y_outer);

  // Compute the blur_x per tile of the blur_y
  blur_x.compute_at(blur_y, x_outer);

  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

void blur_mixed(Buffer<float>& input) {
  Var x("x"), y("y");

  Expr clamped_x = clamp(x, 0, input.width() - 1);
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  Func input_clamped;
  input_clamped(x, y) = input(clamped_x, clamped_y);

  Func blur_x("blur x");
  Func blur_y("blur y");
  blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                  input_clamped(x + 2, y)) /
                 3.0f;
  blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

  Var x_inner, y_inner;

  blur_y.tile(x, y, x_inner, y_inner, 8, 8);
  blur_y.parallel(y);
  blur_y.vectorize(x_inner, 8);
  blur_x.compute_at(blur_y, x);
  blur_x.vectorize(x, 8);

  Timer t1("1 original");
  Buffer<float> output = blur_y.realize(WIDTH, HEIGHT);
  t1.stop();
}

int main() {
  Buffer<float> input(WIDTH, HEIGHT);
  for (int x = 0; x < WIDTH; ++x) {
    for (int y = 0; y < HEIGHT; ++y) {
      input(x, y) = x * WIDTH + y;
    }
  }

  // blur_original(input);

  // blur_x_root(input);

  // blur_x_at_y(input);

  // blur_x_store(input);

  // blur_x_at_x(input);

  blur_tile(input);

  blur_mixed(input);

  // for (int j = 0; j < 4; j++) {
  //   for (int i = 0; i < 4; i++) {
  //     // We can access a pixel of an Buffer object using similar
  //     // syntax to defining and using functions.
  //     printf("%f, ", output(i, j));
  //   }
  //   printf("\n");
  // }

  // printf("Success!\n");

  return 0;
}