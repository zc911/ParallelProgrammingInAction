g++ halide_test.cpp -I/home/chenzhen/Workspace/hpc/halide_build/include -L/home/chenzhen/Workspace/hpc/halide_build/lib -lHalide -lpthread -ldl -std=c++11 -fopenmp -msse2 -msse3 -g -o halide_test 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenzhen/Workspace/hpc/halide_build/lib
./halide_test
