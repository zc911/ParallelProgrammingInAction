export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
/usr/local/cuda/bin/nvcc -Wno-deprecated-gpu-targets -I/usr/local/cuda/include gpu.cu -std=c++11 -o gpu && ./gpu