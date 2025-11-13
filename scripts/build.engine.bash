#!/bin/bash

## Build options
# optShapes="--optShapes=input:1x3x256x256" builds with dynamic shapes using a profile with the opt shape. 
# note that min and max shape must be provided, unless we provide optShapes then defaults to optShapes for min and max.
# inputIOFormats="--inputIOFormats=fp32:chw" builds with the input in fp32 format, is the default which we will leave for now. 
# outputIOFormats="--outputIOFormats=fp32:chw"
## The are optimisation option for the inputIOFormats and outputIOFormats on the memory layout but we use deafult above.
## memory optimisation option that i'm not using for now.

# --fp16, allows the use --fp16 half precision in addtion to the default fp32 precision.
# --stronglyTyped, is by default off,  we leave it off for now, but since only --fp32 is on will only use fp32.

## Note we can't set the optShapes as our model is not dynamic.



exec trtexec --onnx=/home/tom/models/carbot_v1/$1.onnx --saveEngine=/home/tom/models/carbot_v1/$1.engine
