export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
/usr/local/cuda-10.1/bin/nvcc -Wno-deprecated-gpu-targets -I/usr/local/cuda-10.1/include gpu.cu -std=c++11 -O3 -o gpu && ./gpu