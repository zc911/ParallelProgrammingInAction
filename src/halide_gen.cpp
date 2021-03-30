#include "Halide.h"

using namespace Halide;

class BlurGenerator : public Halide::Generator<BlurGenerator> {
 public:
  Input<Buffer<float>> input = {"input", 2};
  Output<Buffer<float>> output = {"output", 2};

  void generate() {
    Var x("x"), y("y"), x_inner("x_inner"), y_inner("y_inner");
    Func blur_x("blur_x");
    Expr clamped_x = clamp(x, 0, input.width() - 1);
    Expr clamped_y = clamp(y, 0, input.height() - 1);
    Func input_clamped;
    input_clamped(x, y) = input(clamped_x, clamped_y);
    blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                    input_clamped(x + 2, y)) /
                   3.0f;
    output(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

    output.tile(x, y, x_inner, y_inner, 8, 8);
    output.parallel(y);
    output.vectorize(y, 12);
    blur_x.store_at(output, x);
    blur_x.compute_at(output, x);
    blur_x.vectorize(x, 12);
  }
};

class AutoBlurGenerator : public Halide::Generator<AutoBlurGenerator> {
 public:
  Input<Buffer<float>> input = {"input", 2};
  Output<Buffer<float>> output = {"output", 2};
  Func blur_x;
  Var x, y, x_inner, y_inner;

  void generate() {
    Expr clamped_x = clamp(x, 0, input.width() - 1);
    Expr clamped_y = clamp(y, 0, input.height() - 1);
    Func input_clamped;
    input_clamped(x, y) = input(clamped_x, clamped_y);
    blur_x(x, y) = (input_clamped(x, y) + input_clamped(x + 1, y) +
                    input_clamped(x + 2, y)) /
                   3.0f;
    output(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;
  }

  void schedule() {
    if (auto_schedule) {
      input.set_estimates({{8192, 8192}, {4096, 4096}});
      output.set_estimates({{8192, 8192}, {4096, 4096}});
    } else {
      output.tile(x, y, x_inner, y_inner, 8, 8);
      output.parallel(y);
      output.vectorize(y, 12);
      blur_x.store_at(output, x);
      blur_x.compute_at(output, x);
      blur_x.vectorize(x, 12);
    }
  }
};

// Register our generator:
HALIDE_REGISTER_GENERATOR(BlurGenerator, blur_mat)
HALIDE_REGISTER_GENERATOR(AutoBlurGenerator, auto_blur_mat)