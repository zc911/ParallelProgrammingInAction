#include <chrono>
#include <iostream>
#include <string>

class Timer {
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> d_start;
  std::string _timer_name;
  double _ms;
  double _cmp_ms = 0;

 public:
  Timer(std::string timer_name) {
    d_start = std::chrono::high_resolution_clock::now();
    _timer_name = timer_name;
  }

  Timer(std::string timer_name, double cmp_ms) {
    d_start = std::chrono::high_resolution_clock::now();
    _timer_name = timer_name;
    _cmp_ms = cmp_ms;
  }

  ~Timer() {}

  void stop() {
    auto d_end = std::chrono::high_resolution_clock::now();
    auto _start =
        std::chrono::time_point_cast<std::chrono::microseconds>(d_start)
            .time_since_epoch()
            .count();
    auto _end = std::chrono::time_point_cast<std::chrono::microseconds>(d_end)
                    .time_since_epoch()
                    .count();
    auto duration = _end - _start;
    _ms = duration * 0.001;

    std::cout << _timer_name << " time cost: " << _ms << "ms ";
    if (_cmp_ms != 0) std::cout << "speedup: " << _cmp_ms / _ms << "x";
    std::cout << "\n";
  }

  double get() { return _ms; }
};