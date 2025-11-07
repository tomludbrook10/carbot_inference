#!/bin/bash


g++ main.cpp $(pkg-config --cflags --libs gstreamer-1.0) \
   -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes/ \
   -I /usr/local/cuda/include/ \
   -I /usr/include/aarch64-linux-gnu/ \
   -I /opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvdspreprocess/include/ \
   -L /opt/nvidia/deepstream/deepstream-7.1/lib/ \
   -L /usr/local/cuda/lib64/ \
   -L /usr/lib/aarch64-linux-gnu/ \
   -lnvinfer \
   -lnvdsgst_meta -lnvds_meta -lcudart -o main