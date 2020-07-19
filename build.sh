mkdir build 
 cd build
cmake -D CMAKE_PREFIX_PATH=/home/ubuntu/local/libtorch/ \
-D CMAKE_BUILD_TYPE=RELEASE \
..
