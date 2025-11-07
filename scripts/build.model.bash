#!/bin/bash

# cuda complier.

nvcc -o run_model run_model.cpp -lnvinfer -lnvonnxparser -lcudart

g++ run_model.cpp $(pkg-config --cflags --libs gstreamer-1.0) \
   -I /usr/local/cuda/include/ \
   -L /usr/local/cuda/lib64/ \
   -lnvinfer -lcudart -o run_model