# build the code generator
g++ halide_gen.cpp /home/chenzhen/Workspace/hpc/halide_build/distrib/tools/GenGen.cpp \
-I/home/chenzhen/Workspace/hpc/halide_build/include -L/home/chenzhen/Workspace/hpc/halide_build/lib \
-lHalide -lpthread -ldl -std=c++11 -fopenmp -msse2 -msse3 -fno-rtti -O3 -o blur_generator

# generate the header file and library
# Set up LD_LIBRARY_PATH so that we can find libHalide.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/chenzhen/Workspace/hpc/halide_build/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/home/chenzhen/Workspace/hpc/halide_build/lib


# First let's compile the first generator for the host system:
./blur_generator -o . -g blur_mat target=host auto_schedule=false
./blur_generator -o . -g auto_blur_mat -f auto_blur_mat -e static_library,h,schedule \
-p /home/chenzhen/Workspace/hpc/halide_build/distrib/lib/libautoschedule_adams2019.so  \
target=host auto_schedule=true

# call the library
g++ halide_gen_main.cpp blur_mat.a auto_blur_mat.a -I/home/chenzhen/Workspace/hpc/src \
-I/home/chenzhen/Workspace/hpc/halide_build/include -L/home/chenzhen/Workspace/hpc/halide_build/lib \
-lHalide  -lpthread -ldl -std=c++11 -fopenmp -O3 -o halide_generator_main
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenzhen/Workspace/hpc/halide_build/lib
./halide_generator_main