module load cmake/3.29.8 openmpi/5.0.5
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release
make -j
