# build the code generator
g++ halide_gen.cpp /home/chenzhen/Workspace/hpc/halide_build/distrib/tools/GenGen.cpp  -I/home/chenzhen/Workspace/hpc/halide_build/include -L/home/chenzhen/Workspace/hpc/halide_build/lib -lHalide -lpthread -ldl -std=c++11 -fopenmp -msse2 -msse3 -fno-rtti -O3 -o blur_generator

# generate the header file and library
chmod +x halide_gen_usage.sh && ./halide_gen_usage.sh

# call the library
g++ halide_gen_main.cpp blur_mat.a auto_blur_mat.a -I/home/chenzhen/Workspace/hpc/src -I/home/chenzhen/Workspace/hpc/halide_build/include -L/home/chenzhen/Workspace/hpc/halide_build/lib -lHalide  -lpthread -ldl -std=c++11 -fopenmp -O3 -o halide_generator_main
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenzhen/Workspace/hpc/halide_build/lib
./halide_generator_main