#!/bin/bash

# cuda complier.

g++ test/run_model.cpp \
   -I /usr/local/cuda/include/ \
   -I /usr/include/aarch64-linux-gnu/ \
   -L /usr/local/cuda/lib64/ \
   -L /usr/lib/aarch64-linux-gnu/ \
   -lnvinfer -lcudart -o main